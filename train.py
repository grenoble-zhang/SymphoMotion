"""
Training script for PCD Controller (Camera Control).
Based on Uni3C inference pipeline, trains the ControlNet branch
to learn camera trajectory control for video generation.
"""
import argparse
import csv
import gc
import json
import logging
import os
import warnings

# Suppress noisy warnings before any other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")
warnings.filterwarnings("ignore", message=".*was not found in config.*")
import math
import os
from pathlib import Path
from datetime import timedelta

import torch
import transformers
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
    InitProcessGroupKwargs,
    DistributedType,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image
from omegaconf import OmegaConf

import diffusers
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from src.models.pcd_controller import PCDController
from src.pcd_dataset import PCDDataset, custom_collate_fn
from src.pipelines.pipeline_pcd import PCDControllerPipeline
from src.dataset_from_npz import load_dataset as load_validation_dataset

check_min_version("0.31.0.dev0")

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description="Training script for SymphoMotion camera/object control.")

    parser.add_argument("--task", type=str, default="train")

    # Dataset setting
    parser.add_argument("--csv_path", type=str, default=None, required=True,
                        help="Path to CSV file containing sample paths in 'path' column.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--max_area", type=int, default=480 * 832, help="Maximum area for resizing.")
    parser.add_argument("--use_camera_embedding", action="store_true", default=True,
                        help="Whether to use camera embedding.")
    parser.add_argument("--use_object_prompt", action="store_true", default=False,
                        help="Whether to use object-specific prompts from prompt-didi.json.")

    # Model setting
    parser.add_argument("--pretrained_model_path", type=str, default=None, required=True,
                        help="Path to pretrained Wan2.1-I2V model.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to config file for controlnet configuration.")
    parser.add_argument("--controlnet_path", type=str, default=None,
                        help="Path to pretrained controlnet weights.")
    parser.add_argument("--output_dir", type=str, default="outputs/symphomotion",
                        help="The output directory where checkpoints will be written.")

    # Controller setting
    parser.add_argument("--save_checkpoint_postfix", type=str, default="", required=False)

    # Validation parameters
    parser.add_argument("--use_log_validation", action="store_true", default=False,
                        help="Whether to run validation at each checkpoint.")
    parser.add_argument("--validation_csv_path", type=str, default=None,
                        help="CSV file with 'id,path' columns for validation samples.")
    parser.add_argument("--num_validation_samples", type=int, default=4)
    parser.add_argument("--validation_guidance_scales", type=float, nargs='+', default=[5.0])
    parser.add_argument("--validation_inference_steps", type=int, default=40)
    parser.add_argument("--validation_fps", type=int, default=16)

    # Checkpoint settings
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from.")

    # Weighting scheme
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])

    # Training setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_architecture", type=str, default="controller_only",
                        choices=["full", "controller_only"])
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["adam", "adamw", "prodigy", "Adam", "AdamW"])
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # Other
    parser.add_argument("--tracker_name", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--nccl_timeout", type=int, default=7200)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # Object trajectory injection (Stage 2)
    parser.add_argument("--obj_cross_attn_interval", type=int, default=2,
                        help="[DEPRECATED] This parameter is kept for compatibility but no longer controls injection frequency. "
                             "Object features are now injected in the first 20 layers (0-19) regardless of this value. "
                             "Set to >0 to enable object injection, 0 to disable.")
    parser.add_argument("--obj_scale", type=float, default=1.0,
                        help="Scale factor for object trajectory injection.")
    parser.add_argument("--obj_traj_mid_dim", type=int, default=512,
                        help="Intermediate dimension for trajectory point projection (PointNet output dim).")
    parser.add_argument("--max_entities", type=int, default=2,
                        help="Maximum number of object entities.")
    parser.add_argument("--normalize_object_to_first_frame", action=argparse.BooleanOptionalAction, default=True,
                        help="Transform object trajectories from per-frame camera coordinates into the first-frame camera coordinate system.")
    parser.add_argument("--max_text_tokens", type=int, default=50,
                        help="Maximum text tokens per entity for pairwise fusion (default 50).")
    parser.add_argument("--obj_injector_path", type=str, default=None,
                        help="Path to pretrained object injector weights (for resuming Stage 2).")

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_controlnet_components(model, save_path, accelerator):
    """Extract and save ControlNet-related components from the model."""
    unwrapped_model = accelerator.unwrap_model(model)
    model_state_dict = unwrapped_model.state_dict()

    controlnet_state_dict = {}
    for key, value in model_state_dict.items():
        if "controlnet" in key:
            controlnet_state_dict[key] = value

    torch.save(controlnet_state_dict, save_path)


def save_checkpoint(model, save_dir, accelerator, save_object_injector=False, save_controlnet=True):
    """Save ControlNet and optionally Object Injector components (rank 0 only)."""
    # Only rank 0 should save to avoid NFS race conditions
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)

    # Save controlnet only if trainable
    if save_controlnet:
        controlnet_save_path = os.path.join(save_dir, "controlnet.pth")
        save_controlnet_components(model, controlnet_save_path, accelerator)

    # Save object injector if applicable
    if save_object_injector:
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'obj_traj_encoder'):
            obj_injector_path = os.path.join(save_dir, "object_injector.pth")
            unwrapped_model.save_object_injector(obj_injector_path)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    if use_deepspeed:
        from accelerate.utils import DummyOptim
        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    optimizer_lower = args.optimizer.lower()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`.")
        if optimizer_lower == "adamw":
            optimizer_class = bnb.optim.AdamW8bit
        elif optimizer_lower == "adam":
            optimizer_class = bnb.optim.Adam8bit
        else:
            logger.warning(f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW8bit")
            optimizer_class = bnb.optim.AdamW8bit
    else:
        if optimizer_lower == "adamw":
            optimizer_class = torch.optim.AdamW
        elif optimizer_lower == "adam":
            optimizer_class = torch.optim.Adam
        else:
            logger.warning(f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW")
            optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Batch preprocessing (reuses pipeline methods to guarantee alignment)
# ---------------------------------------------------------------------------

def preprocess_batch(batch, pipeline, device, dtype, num_frames=81, encode_object_prompts=False):
    """
    Preprocess a raw batch from PCDDataset into training-ready tensors.
    Calls pipeline.encode_prompt / encode_image / encode_condition /
    encode_render_latent directly so that training and inference share
    the exact same preprocessing code path.
    """
    with torch.no_grad():
        # ---- 1. Encode text prompts (reuse pipeline's encode_prompt) ----
        # bf16
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=batch['prompt'],
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=device,
            dtype=dtype,
        )

        # ---- 2. Encode reference images with CLIP (reuse pipeline's encode_image) ----
        images = batch['image']  # list of tensors [3, H, W] in [0, 1]
        pil_images = [
            Image.fromarray(
                (x.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            )
            for x in images
        ]
        # bf16
        image_embeds = pipeline.encode_image(pil_images, device).to(dtype=dtype)

        # ---- 3. Encode render video (reuse pipeline's encode_render_latent) ----
        render_video = batch['render_video'].to(device, dtype=torch.float32)
        render_latent = pipeline.encode_render_latent(render_video, device)

        # ---- 4. Target video latent ----
        # Encode ground-truth video through VAE as the denoising target for flow matching
        gt_video = batch['gt_video'].to(device, dtype=torch.float32)
        video_latent = pipeline.encode_render_latent(gt_video, device)

        # ---- 5. Create condition (reuse pipeline's encode_condition) ----
        # Convert images from [0,1] to [-1,1] matching pipeline's video_processor.preprocess
        image_tensor = torch.stack(images).to(device, dtype=torch.float32)
        image_preprocessed = image_tensor * 2.0 - 1.0
        H, W = image_preprocessed.shape[2], image_preprocessed.shape[3]
        condition = pipeline.encode_condition(
            image_preprocessed, num_frames, H, W, device
        )

        # ---- 6. Render mask at pixel resolution ----
        # Keep fp32 to match inference pipeline (pipeline_pcd.py passes these without dtype conversion)
        render_mask = batch['render_mask'].to(device)

        # ---- 7. Camera embedding at pixel resolution ----
        camera_embedding = None
        if batch['camera_embedding'] is not None:
            camera_embedding = batch['camera_embedding'].to(device)

        # ---- 8. Encode object prompts with T5 (Stage 2, 3DTrajMaster style) ----
        obj_text_embeds = None
        obj_text_masks = None
        if encode_object_prompts and batch.get('object_prompts_list') is not None:
            prompts_lists = batch['object_prompts_list']  # list[list[str]], outer=B, inner=max_entities
            B = len(prompts_lists)
            max_ent = len(prompts_lists[0])

            entity_embeds = []
            entity_masks = []
            for ent_idx in range(max_ent):
                # Collect this entity's prompt from each sample in the batch
                ent_prompts = [prompts_lists[b][ent_idx] for b in range(B)]

                # Encode with T5 - need to get attention mask too
                # Use tokenizer directly to get mask
                text_inputs = pipeline.tokenizer(
                    ent_prompts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                attention_mask = text_inputs.attention_mask.to(device)  # [B, 77]

                # Get embeddings
                ent_emb = pipeline.text_encoder(text_input_ids, attention_mask).last_hidden_state  # [B, 77, 4096]
                ent_emb = ent_emb.to(dtype=torch.float32)
                attention_mask = attention_mask.to(dtype=torch.float32)

                entity_embeds.append(ent_emb)
                entity_masks.append(attention_mask)

            # Stack entities: [B, max_entities, 77, 4096] and [B, max_entities, 77]
            obj_text_embeds = torch.stack(entity_embeds, dim=1)
            obj_text_masks = torch.stack(entity_masks, dim=1)

    result = {
        'video_latent': video_latent,       # [B, 16, F_lat, H_lat, W_lat] fp32
        'render_latent': render_latent,     # [B, 16, F_lat, H_lat, W_lat] fp32
        'condition': condition,             # [B, 20, F_lat, H_lat, W_lat] fp32
        'render_mask': render_mask,         # [B, 1, F, H, W] pixel res fp32
        'camera_embedding': camera_embedding,  # [B, 6, F, H, W] pixel res or None fp32
        'prompt_embeds': prompt_embeds,     # [B, seq_len, 4096] bf16
        'image_embeds': image_embeds,       # [B, 257, hidden_dim] bf16
    }
    if obj_text_embeds is not None:
        result['obj_text_embeds'] = obj_text_embeds  # [B, max_entities, 77, 4096] fp32
        result['obj_text_masks'] = obj_text_masks    # [B, max_entities, 77] fp32
    return result


# ---------------------------------------------------------------------------
# Validation (following reference inference pattern)
# ---------------------------------------------------------------------------

def log_validation(args, transformer, accelerator, global_step, cfg,
                   tokenizer, text_encoder, image_encoder, image_processor, vae):
    """
    Run validation in parallel across all ranks.
    Each rank processes a subset of validation samples, avoiding NCCL timeouts.
    Works for both single-node and multi-node setups.
    """
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    logger.info(f"[Rank {rank}/{world_size}] Running validation at step {global_step}...")
    torch.cuda.empty_cache()
    transformer.eval()

    # Move encoder models to GPU for validation
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    image_encoder.to(accelerator.device)

    local_video_infos = []  # Initialize early for all code paths

    try:
        # ---- Read validation CSV (all ranks read the same list) ----
        if not args.validation_csv_path or not os.path.exists(args.validation_csv_path):
            logger.warning("Validation CSV not found, skipping validation")
            # Don't return early - must reach barrier at end
        else:
            path_list = []
            with open(args.validation_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path = row['path'].strip()
                    if path and os.path.exists(path):
                        path_list.append(path)
                    elif path and rank == 0:
                        logger.warning(f"Validation path not found, skipping: {path}")

            if not path_list:
                if rank == 0:
                    logger.warning("No valid paths in validation CSV")
                # Don't return early - must reach barrier at end
            else:
                path_list = path_list[:args.num_validation_samples]
                logger.info(f"[Rank {rank}] Validation: {len(path_list)} total samples from {args.validation_csv_path}")

                # Skip validation if insufficient samples to avoid NCCL timeout
                if len(path_list) < world_size:
                    if rank == 0:
                        logger.warning(
                            f"Skipping validation: only {len(path_list)} samples for {world_size} ranks. "
                            f"Validation requires at least {world_size} samples to avoid NCCL watchdog timeout. "
                            f"Please increase --num_validation_samples to at least {world_size}."
                        )
                    # All ranks must skip together to avoid barrier deadlock
                    transformer.train()
                    return

                # ---- Split samples across ranks ----
                local_indices = list(range(rank, len(path_list), world_size))
                local_paths = [path_list[i] for i in local_indices]
                logger.info(f"[Rank {rank}] Assigned {len(local_paths)} validation samples: "
                             f"indices {local_indices}")
                if not local_paths:
                    logger.info(f"[Rank {rank}] No validation samples assigned on this rank")

                # ---- Create pipeline (matching reference code) ----
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    args.pretrained_model_path, subfolder="scheduler"
                )
                pipeline = PCDControllerPipeline(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    image_encoder=image_encoder,
                    image_processor=image_processor,
                    transformer=unwrapped_transformer,
                    vae=vae,
                    scheduler=scheduler,
                )
                pipeline.set_progress_bar_config(disable=True)
                device = accelerator.device

                use_camera_embedding = (
                    cfg.get("camera_embedding", False) if cfg else args.use_camera_embedding
                )

                # ---- Each rank processes its own subset ----
                with torch.no_grad():
                    for local_idx, (global_idx, base_path) in enumerate(zip(local_indices, local_paths)):
                        base_name = os.path.basename(base_path)
                        logger.info(f"[Rank {rank}] Validation {local_idx + 1}/{len(local_paths)}: {base_name}")

                        reference_image = os.path.join(base_path, "first_image.png")
                        render_path = os.path.join(base_path, "render_output")
                        prompt_path = os.path.join(base_path, "full_prompt.json")

                        # Check required files
                        if not os.path.exists(reference_image):
                            logger.warning(f"Skip {base_path}: first_image.png not found")
                            continue
                        if not os.path.exists(render_path):
                            logger.warning(f"Skip {base_path}: render_output/ not found")
                            continue
                        if not os.path.exists(prompt_path):
                            logger.warning(f"Skip {base_path}: full_prompt.json not found")
                            continue

                        try:
                            # Read prompt (matching reference)
                            with open(prompt_path, 'r', encoding='utf-8') as f:
                                prompt_data = json.load(f)
                                prompt = prompt_data.get("full_prompt", "")

                            # Load data using load_dataset (matching reference exactly)
                            logger.info(f"[Rank {rank}] Loading validation data for {base_name}")
                            (image, render_video, render_mask, camera_embedding, height, width,
                             camera_3d_preds, object_prompts_list, num_entities) = (
                                load_validation_dataset(
                                    reference_image=reference_image,
                                    render_path=render_path,
                                    nframe=args.num_frames,
                                    max_area=args.max_area,
                                    pipe=pipeline,
                                    use_camera_embedding=use_camera_embedding,
                                    device=device,
                                    sp_degree=1,
                                    logger=logger,
                                    use_object_prompt=args.use_object_prompt,
                                    max_entities=args.max_entities,
                                    normalize_object_to_first_frame=args.normalize_object_to_first_frame,
                                )
                            )
                            logger.info(f"[Rank {rank}] Loaded validation data for {base_name}")

                            # Move encoders to CPU after encoding to free GPU memory
                            vae.to('cpu')
                            text_encoder.to('cpu')
                            image_encoder.to('cpu')
                            torch.cuda.empty_cache()

                            # Prepare object trajectory inputs (if enabled)
                            if args.use_object_prompt and camera_3d_preds is not None:
                                camera_3d_preds_input = camera_3d_preds.unsqueeze(0).to(dtype=torch.float32, device=accelerator.device)  # [1, max_entities, 81, 500, 3]
                                object_prompts_list_input = [object_prompts_list]  # [[str, str, ...]]
                                num_entities_input = torch.tensor([num_entities], device=accelerator.device)
                            else:
                                camera_3d_preds_input = None
                                object_prompts_list_input = None
                                num_entities_input = None

                            # Move VAE back to GPU for inference
                            # Note: transformer is already on GPU (managed by accelerator)
                            vae.to(accelerator.device)

                            # Debug: verify device placement
                            if hasattr(unwrapped_transformer, 'obj_traj_encoder'):
                                obj_encoder_device = next(unwrapped_transformer.obj_traj_encoder.parameters()).device
                                logger.info(f"[Rank {rank}] obj_traj_encoder device: {obj_encoder_device}")

                            for guidance_scale in args.validation_guidance_scales:
                                gen = torch.Generator(device=accelerator.device)
                                gen.manual_seed(args.seed)

                                logger.info(
                                    f"[Rank {rank}] Starting pipeline inference for {base_name} "
                                    f"with guidance_scale={guidance_scale}"
                                )
                                output = pipeline(
                                    image=image,
                                    render_video=render_video.to(device),
                                    render_mask=render_mask.to(device),
                                    camera_embedding=(
                                        camera_embedding.to(device)
                                        if camera_embedding is not None else None
                                    ),
                                    camera_3d_preds=camera_3d_preds_input,
                                    object_prompts_list=object_prompts_list_input,
                                    num_entities=num_entities_input,
                                    prompt=prompt,
                                    negative_prompt="",
                                    height=height,
                                    width=width,
                                    num_frames=args.num_frames,
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=args.validation_inference_steps,
                                    generator=gen,
                                )
                                logger.info(
                                    f"[Rank {rank}] Finished pipeline inference for {base_name} "
                                    f"with guidance_scale={guidance_scale}"
                                )

                                generated_video = output.frames[0]
                                validation_dir = os.path.join(
                                    args.output_dir, "validation_videos", f"step_{global_step}"
                                )
                                os.makedirs(validation_dir, exist_ok=True)
                                suffix = f"_cfg_{guidance_scale}" if len(args.validation_guidance_scales) > 1 else ""
                                output_path = os.path.join(validation_dir, f"{base_name}{suffix}.mp4")
                                logger.info(f"[Rank {rank}] Exporting validation video to {output_path}")
                                export_to_video(generated_video, output_path, fps=args.validation_fps)
                                logger.info(f"[Rank {rank}] Saved: {output_path}")

                                # Collect video info for wandb upload after all ranks finish
                                local_video_infos.append({
                                    "output_path": output_path,
                                    "key": f"validation/{base_name}{suffix}",
                                    "caption": f"{base_name}: {prompt[:50]}",
                                })

                                # Cleanup per guidance scale
                                del output, generated_video
                                torch.cuda.empty_cache()

                            # Move VAE back to CPU after all guidance scales
                            # Note: Don't move unwrapped_transformer to CPU as it's managed by accelerator
                            vae.to('cpu')
                            torch.cuda.empty_cache()

                            # Cleanup per-sample
                            del image, render_video, render_mask, camera_embedding
                            torch.cuda.empty_cache()

                            # Move encoders back to GPU for next sample
                            # Note: transformer stays on GPU (managed by accelerator)
                            if local_idx + 1 < len(local_paths):
                                vae.to(accelerator.device)
                                text_encoder.to(accelerator.device)
                                image_encoder.to(accelerator.device)

                        except Exception as e:
                            logger.warning(f"[Rank {rank}] Validation failed for {base_path}: {e}")
                            import traceback
                            traceback.print_exc()

                            # Ensure models are moved back to CPU even on failure
                            # Note: Don't move transformer to CPU as it's managed by accelerator
                            try:
                                vae.to('cpu')
                                torch.cuda.empty_cache()
                            except:
                                pass

                            continue

                del pipeline
                torch.cuda.empty_cache()
                gc.collect()


    except Exception as e:
        logger.warning(f"[Rank {rank}] Validation pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    # CRITICAL: All ranks must reach this barrier regardless of success/failure
    # Gather video info from all ranks and upload on rank 0
    logger.info(f"[Rank {rank}] Validation loop finished with {len(local_video_infos)} generated videos")
    torch.cuda.synchronize()
    logger.info(f"[Rank {rank}] Entering accelerator.wait_for_everyone() after validation")
    accelerator.wait_for_everyone()
    logger.info(f"[Rank {rank}] Passed accelerator.wait_for_everyone() after validation")
    if world_size > 1:
        import torch.distributed as dist
        all_video_infos = [None] * world_size
        logger.info(f"[Rank {rank}] Entering dist.all_gather_object for validation video metadata")
        dist.all_gather_object(all_video_infos, local_video_infos)
        logger.info(f"[Rank {rank}] Finished dist.all_gather_object for validation video metadata")
    else:
        all_video_infos = [local_video_infos]

    if rank == 0:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                for rank_infos in all_video_infos:
                    for info in rank_infos:
                        tracker.log({
                            info["key"]: wandb.Video(
                                info["output_path"],
                                fps=args.validation_fps,
                                caption=info["caption"],
                            )
                        }, step=global_step)

    # Ensure encoder models are on CPU after validation
    # (they should already be there from per-sample cleanup)
    vae.to('cpu')
    text_encoder.to('cpu')
    image_encoder.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()

    transformer.train()


# ---------------------------------------------------------------------------
# Flow matching helpers
# ---------------------------------------------------------------------------

def compute_density_for_timestep_sampling(weighting_scheme, batch_size, device):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device)
        u = 1 - u - 1.29 * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device)
    return u


def compute_loss_weighting(weighting_scheme, sigmas):
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas ** -2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas ** 2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device=None):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS.")

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_kwargs = InitProcessGroupKwargs(
        backend="nccl", timeout=timedelta(seconds=args.nccl_timeout)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------
    load_dtype = torch.bfloat16

    cfg = None
    if args.config_path and os.path.exists(args.config_path):
        logger.info(f"Loading config from {args.config_path}")
        cfg = OmegaConf.load(args.config_path)

    # Load transformer (base Wan I2V + ControlNet)
    transformer = PCDController.from_pretrained(
        args.pretrained_model_path,
        subfolder="transformer",
        controlnet_cfg=cfg.controlnet_cfg if cfg else None,
        torch_dtype=load_dtype,
    )
    transformer.build_controlnet(model_path=args.controlnet_path, logger=logger)

    # Build object injector if Stage 2 (use_object_prompt)
    if args.use_object_prompt:
        transformer.build_object_injector(
            obj_cross_attn_interval=args.obj_cross_attn_interval,
            obj_scale=args.obj_scale,
            traj_mid_dim=args.obj_traj_mid_dim,
            max_entities=args.max_entities,
            num_frames=args.num_frames,
            max_text_tokens=args.max_text_tokens,
            obj_injector_path=args.obj_injector_path,
            logger=logger,
        )

    # Load noise scheduler for flow matching training
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_path, subfolder="scheduler"
    )
    # Initialize timesteps and sigmas for training
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device="cpu")

    # -------------------------------------------------------------------
    # Set requires_grad based on training architecture
    # -------------------------------------------------------------------
    if args.use_object_prompt:
        # Stage 2: Freeze everything (base DiT + camera ControlNet), train only object injector
        transformer.requires_grad_(False)
        trainable_param_count = 0
        for name, param in transformer.named_parameters():
            if "obj_traj_encoder" in name or "obj_perceiver_cross_attention" in name:
                param.requires_grad_(True)
                trainable_param_count += param.numel()
        logger.info(f"Stage 2 (object injection): {trainable_param_count:,} trainable parameters")
        logger.info(f"  Frozen: base DiT + camera ControlNet")
    elif args.train_architecture == "full":
        transformer.requires_grad_(True)
        logger.info("Training full model (all parameters)")
    elif args.train_architecture == "controller_only":
        transformer.requires_grad_(False)
        trainable_param_count = 0
        for name, param in transformer.named_parameters():
            if "controlnet" in name.lower():
                param.requires_grad_(True)
                trainable_param_count += param.numel()
        logger.info(f"Controller-only training: {trainable_param_count:,} trainable parameters")

    transformer.to(accelerator.device, dtype=load_dtype)

    # Restore controlnet params to fp32 (they were loaded as fp32 but
    # the .to() call above converted them to load_dtype/bf16)
    for name, param in transformer.named_parameters():
        if "controlnet" in name.lower():
            param.data = param.data.to(torch.float32)

    # Restore object injector params and buffers to fp32 for precision (Stage 2)
    if args.use_object_prompt:
        # Restore parameters (weight, bias) to fp32
        for name, param in transformer.named_parameters():
            if "obj_traj_encoder" in name or "obj_perceiver_cross_attention" in name:
                param.data = param.data.to(torch.float32)
        # Restore buffers (BatchNorm's running_mean, running_var) to fp32
        for name, buffer in transformer.named_buffers():
            if "obj_traj_encoder" in name or "obj_perceiver_cross_attention" in name:
                buffer.data = buffer.data.to(torch.float32)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # -------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": trainable_params, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    args.use_deepspeed = accelerator.state.deepspeed_plugin is not None

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # -------------------------------------------------------------------
    # Load auxiliary models (frozen, for data preprocessing only)
    # Keep on CPU initially; move to GPU on demand to reduce peak VRAM.
    # -------------------------------------------------------------------
    logger.info("Loading auxiliary models for data preprocessing...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder", torch_dtype=load_dtype
    ).eval()

    image_processor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_path, subfolder="image_processor"
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        args.pretrained_model_path, subfolder="image_encoder", torch_dtype=torch.float32
    ).eval()

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float32
    ).eval()

    # Enable VAE tiling to reduce peak memory during 3D encoding
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
        logger.info("VAE tiling enabled for memory efficiency")

    # -------------------------------------------------------------------
    # Create pipeline for preprocessing (shares encode_* methods with inference)
    # Only used for its helper methods, NOT for __call__
    # -------------------------------------------------------------------
    preprocess_pipeline = PCDControllerPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_processor=image_processor,
        transformer=transformer,
        vae=vae,
        scheduler=noise_scheduler,
    )

    # -------------------------------------------------------------------
    # Dataset and DataLoader
    # -------------------------------------------------------------------
    train_dataset = PCDDataset(
        csv_path=args.csv_path,
        num_frames=args.num_frames,
        max_area=args.max_area,
        use_camera_embedding=args.use_camera_embedding,
        use_object_prompt=args.use_object_prompt,
        max_entities=args.max_entities,
        device="cpu",
        normalize_object_to_first_frame=args.normalize_object_to_first_frame,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.dataloader_num_workers != 0 else None,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
    )

    # -------------------------------------------------------------------
    # LR Scheduler
    # -------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # -------------------------------------------------------------------
    # Prepare with accelerator
    # -------------------------------------------------------------------
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # -------------------------------------------------------------------
    # Initialize trackers
    # -------------------------------------------------------------------
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "symphomotion"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # -------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if os.path.isdir(checkpoint_path):
            # Load controlnet weights
            controlnet_pth = os.path.join(checkpoint_path, "controlnet.pth")
            if os.path.exists(controlnet_pth):
                unwrapped = accelerator.unwrap_model(transformer)
                state_dict = torch.load(controlnet_pth, map_location="cpu")
                missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
                logger.info(f"Resumed controlnet from {controlnet_pth}")
                if unexpected:
                    logger.info(f"Unexpected keys: {unexpected}")
            # Load object injector weights (Stage 2)
            obj_injector_pth = os.path.join(checkpoint_path, "object_injector.pth")
            if os.path.exists(obj_injector_pth) and hasattr(accelerator.unwrap_model(transformer), 'obj_traj_encoder'):
                accelerator.unwrap_model(transformer).load_object_injector(obj_injector_pth, logger=logger)
            # Try to extract step number from directory name
            dir_name = os.path.basename(checkpoint_path)
            if dir_name.startswith("checkpoint-"):
                try:
                    global_step = int(dir_name.split("-")[1])
                    first_epoch = global_step // num_update_steps_per_epoch
                    logger.info(f"Resuming from step {global_step}, epoch {first_epoch}")
                except (ValueError, IndexError):
                    pass

    # -------------------------------------------------------------------
    # Training info
    # -------------------------------------------------------------------
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    num_trainable_parameters = sum(p.numel() for p in trainable_params if p.requires_grad)

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters:,}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    warned_loss_without_grad = False

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            free_memory()
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                # ---- Stage 1: Move encoder models to GPU and preprocess ----
                free_memory()
                vae.to(accelerator.device)
                text_encoder.to(accelerator.device)
                image_encoder.to(accelerator.device)

                processed = preprocess_batch(
                    batch=batch,
                    pipeline=preprocess_pipeline,
                    device=accelerator.device,
                    dtype=load_dtype,
                    num_frames=args.num_frames,
                    encode_object_prompts=args.use_object_prompt,
                )

                video_latents = processed['video_latent']
                render_latents = processed['render_latent']
                condition = processed['condition']
                prompt_embeds = processed['prompt_embeds']
                image_embeds = processed['image_embeds']
                render_masks = processed['render_mask']
                camera_embeddings = processed['camera_embedding']
                obj_text_embeds = processed.get('obj_text_embeds', None)
                obj_text_masks = processed.get('obj_text_masks', None)
                del processed

                # ---- Stage 2: Offload encoder models to CPU to free VRAM ----
                vae.to('cpu')
                text_encoder.to('cpu')
                image_encoder.to('cpu')
                free_memory()

                # ---- Encode object trajectories (Stage 2 only, after encoder offload) ----
                obj_embeds = None
                if args.use_object_prompt and obj_text_embeds is not None and batch.get('camera_3d_preds') is not None:
                    camera_3d_preds = batch['camera_3d_preds'].to(accelerator.device, dtype=torch.float32)
                    num_entities = batch['num_entities'].to(accelerator.device)

                    unwrapped = accelerator.unwrap_model(transformer)
                    obj_embeds = unwrapped.obj_traj_encoder(
                        camera_3d_preds=camera_3d_preds,
                        obj_text_embeds=obj_text_embeds.to(accelerator.device),
                        obj_text_masks=obj_text_masks.to(accelerator.device),
                        num_entities=num_entities,
                    )

                batch_size = video_latents.shape[0]

                # ---- Sample noise and timesteps (flow matching) ----
                noise = torch.randn_like(video_latents, device=accelerator.device)

                u = compute_density_for_timestep_sampling(
                    args.weighting_scheme, batch_size, accelerator.device
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                indices = indices.clamp(0, len(noise_scheduler.timesteps) - 1)

                timesteps = noise_scheduler.timesteps[indices.cpu()].to(device=video_latents.device)

                # ---- Add noise (flow matching interpolation) ----
                sigmas = get_sigmas(
                    noise_scheduler, timesteps,
                    n_dim=video_latents.ndim,
                    dtype=video_latents.dtype,
                    device=accelerator.device,
                )
                noisy_latents = (1.0 - sigmas) * video_latents + sigmas * noise
                target = noise - video_latents  # velocity prediction target

                # ---- Construct model input ----
                # [B, 16, F_lat, H_lat, W_lat] + [B, 20, F_lat, H_lat, W_lat]
                # = [B, 36, F_lat, H_lat, W_lat] matching I2V transformer's in_channels
                latent_model_input = torch.cat([noisy_latents, condition], dim=1).to(dtype=load_dtype)

                # ---- Stage 3: Forward pass (only transformer on GPU) ----
                model_output = transformer(
                    hidden_states=latent_model_input,
                    render_latent=render_latents,
                    render_mask=render_masks,
                    camera_embedding=camera_embeddings,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    obj_embeds=obj_embeds,
                    return_dict=False,
                )[0]

                # ---- Compute loss ----
                weighting = compute_loss_weighting(args.weighting_scheme, sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_output.float() - target.float()) ** 2)
                    .reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Ensure loss is a scalar for DeepSpeed
                if loss.dim() > 0:
                    loss = loss.mean()

                # Check for NaN/Inf and skip this batch if found
                if not torch.isfinite(loss):
                    sample_id = batch.get("sample_id", ["unknown"])[0] if "sample_id" in batch else "unknown"
                    logger.warning(
                        f"[Rank {accelerator.process_index}] Skipping batch with non-finite loss: {loss.item()} "
                        f"(sample={sample_id})"
                    )
                    # Clean up tensors and skip backward
                    del model_output, latent_model_input, noisy_latents, noise
                    del video_latents, render_latents, condition, target
                    del prompt_embeds, image_embeds, render_masks, camera_embeddings
                    del sigmas, weighting, timesteps
                    if obj_embeds is not None:
                        del obj_embeds
                    if 'camera_3d_preds' in locals():
                        del camera_3d_preds
                    if 'num_entities' in locals():
                        del num_entities
                    if obj_text_embeds is not None:
                        del obj_text_embeds
                    free_memory()
                    continue

                if loss.grad_fn is None:
                    # Keep the backward graph valid even when a bad batch has no trainable path.
                    # This avoids DeepSpeed assertion failures and cross-rank desync.
                    dummy = torch.zeros((), device=accelerator.device, dtype=loss.dtype)
                    for p in trainable_params:
                        if p.requires_grad:
                            dummy = dummy + p.sum() * 0.0
                    if dummy.grad_fn is None:
                        raise RuntimeError("Loss has no grad_fn and no trainable parameters are connected.")
                    loss = loss + dummy
                    if not warned_loss_without_grad:
                        sample_id = batch["sample_id"][0] if len(batch["sample_id"]) > 0 else "unknown"
                        logger.warning(
                            f"[Rank {accelerator.process_index}] Encountered batch with loss.grad_fn=None "
                            f"(sample={sample_id}). Applied zero-gradient fallback for distributed safety."
                        )
                        warned_loss_without_grad = True

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            # ---- Stage 4: Explicitly release all intermediate tensors ----
            # Delete largest tensors first for better memory management
            loss = loss.detach()
            del model_output, latent_model_input, noisy_latents, noise
            del video_latents, render_latents, condition, target
            del prompt_embeds, image_embeds, render_masks, camera_embeddings
            del sigmas, weighting, timesteps
            # Delete object-related tensors early if present
            if obj_embeds is not None:
                del obj_embeds
            if 'camera_3d_preds' in locals():
                del camera_3d_preds
            if 'num_entities' in locals():
                del num_entities
            if obj_text_embeds is not None:
                del obj_text_embeds
            free_memory()

            # ---- Logging and checkpointing ----
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                should_save = (
                    accelerator.distributed_type == DistributedType.DEEPSPEED
                    or accelerator.is_main_process
                )
                if should_save and global_step % args.checkpointing_steps == 0:
                    # Handle checkpoint limit
                    if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                        checkpoints = [
                            d for d in os.listdir(args.output_dir)
                            if d.startswith("checkpoint-")
                        ]
                        # Sort checkpoints, handling both numeric and non-numeric suffixes
                        def get_checkpoint_number(x):
                            try:
                                return int(x.split("-")[1])
                            except (ValueError, IndexError):
                                return float('inf')  # Put non-numeric checkpoints at the end
                        checkpoints = sorted(checkpoints, key=get_checkpoint_number)
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for ckpt in checkpoints[:num_to_remove]:
                                import shutil
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))
                                logger.info(f"Removed old checkpoint: {ckpt}")

                    save_path = os.path.join(
                        args.output_dir,
                        f"checkpoint-{global_step}{args.save_checkpoint_postfix}"
                    )
                    # Determine whether to save controlnet based on training mode
                    # - Stage 2 (use_object_prompt): controlnet is frozen, don't save
                    # - Controller-only or full training: controlnet is trainable, save it
                    should_save_controlnet = not args.use_object_prompt
                    save_checkpoint(transformer, save_path, accelerator,
                                    save_object_injector=args.use_object_prompt,
                                    save_controlnet=should_save_controlnet)
                    # Wait for rank 0 to finish saving before all ranks proceed
                    accelerator.wait_for_everyone()
                    logger.info(f"Saved checkpoint at step {global_step}: {save_path}")

                # Run validation — all ranks participate in parallel
                if global_step % args.checkpointing_steps == 0 and args.use_log_validation and args.validation_csv_path:
                    with torch.no_grad():
                        log_validation(
                            args=args,
                            transformer=transformer,
                            accelerator=accelerator,
                            global_step=global_step,
                            cfg=cfg,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            image_encoder=image_encoder,
                            image_processor=image_processor,
                            vae=vae,
                        )
                    accelerator.wait_for_everyone()
                    # Explicit memory cleanup after validation
                    torch.cuda.empty_cache()
                    transformer.train()

            # Log metrics (only at optimizer steps, not every micro-step)
            if accelerator.sync_gradients:
                try:
                    lr_value = lr_scheduler.get_last_lr()[0]
                except (AttributeError, IndexError):
                    lr_value = args.learning_rate

                loss_value = loss.detach().item()
                logs = {"loss": loss_value, "lr": lr_value, "epoch": epoch}
                progress_bar.set_postfix(**logs)
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step}: loss={loss_value:.6f}, lr={lr_value:.2e}, epoch={epoch}")
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # -------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------
    if args.use_log_validation and args.validation_csv_path:
        if accelerator.is_main_process:
            logger.info("Running final validation...")
        with torch.no_grad():
            log_validation(
                args=args,
                transformer=transformer,
                accelerator=accelerator,
                global_step=global_step,
                cfg=cfg,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                image_encoder=image_encoder,
                image_processor=image_processor,
                vae=vae,
            )
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        save_path = os.path.join(
            args.output_dir, f"checkpoint-final{args.save_checkpoint_postfix}"
        )
        # Determine whether to save controlnet based on training mode
        should_save_controlnet = not args.use_object_prompt
        save_checkpoint(transformer, save_path, accelerator,
                        save_object_injector=args.use_object_prompt,
                        save_controlnet=should_save_controlnet)
        logger.info(f"Training completed. Final model saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
