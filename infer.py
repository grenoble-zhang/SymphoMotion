import argparse
import csv
import gc
import json
import logging
import os
import subprocess
from pathlib import Path

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from src.dataset_from_npz import load_dataset as load_validation_dataset
from src.models.pcd_controller import PCDController
from src.pipelines.pipeline_pcd import PCDControllerPipeline


DEFAULT_VALIDATION_CSV_PATH = str(Path(__file__).resolve().parent / "assets" / "demo.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="CSV-driven SymphoMotion inference script.")
    parser.add_argument("--validation_csv_path", type=str, default=DEFAULT_VALIDATION_CSV_PATH)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, default=None)
    parser.add_argument("--obj_injector_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_area", type=int, default=480 * 832)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_object_prompt", action="store_true", default=False)
    parser.add_argument("--obj_cross_attn_interval", type=int, default=2)
    parser.add_argument("--obj_scale", type=float, default=1.0)
    parser.add_argument("--obj_traj_mid_dim", type=int, default=512)
    parser.add_argument("--max_entities", type=int, default=2)
    parser.add_argument("--max_text_tokens", type=int, default=50)
    parser.add_argument("--save_concat_video", action="store_true", default=False)
    parser.add_argument("--normalize_object_to_first_frame", action=argparse.BooleanOptionalAction, default=True,
                        help="Transform object trajectories from per-frame camera coordinates into the first-frame camera coordinate system.")

    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--use_camera_embedding", dest="use_camera_embedding", action="store_true")
    camera_group.add_argument("--disable_camera_embedding", dest="use_camera_embedding", action="store_false")
    parser.set_defaults(use_camera_embedding=True)

    return parser.parse_args()


def setup_logger(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        force=True,
    )
    return logging.getLogger("infer_validation_csv")


def get_runtime_context():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference script.")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def resolve_controlnet_path(args):
    controlnet_path = args.controlnet_path or os.environ.get("CONTROLNET_PATH")
    if not controlnet_path:
        raise ValueError("Missing ControlNet weights. Provide --controlnet_path or set CONTROLNET_PATH.")
    return controlnet_path


def resolve_obj_injector_path(args, controlnet_path):
    if not args.use_object_prompt:
        return None

    if args.obj_injector_path:
        obj_injector_path = args.obj_injector_path
    else:
        obj_injector_path = str(
            Path(__file__).resolve().parent / "pretrained_checkpoints" / "object_control" / "object_injector.pth"
        )

    if not os.path.exists(obj_injector_path) and os.path.basename(obj_injector_path) != "object_injector.pth":
        raise FileNotFoundError(
            f"use_object_prompt is enabled, but object injector weights were not found: {obj_injector_path}"
        )
    return obj_injector_path


def load_path_list(csv_path, max_samples, logger):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    path_list = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain a 'path' column: {csv_path}")

        for row in reader:
            path = row["path"].strip()
            if not path:
                continue
            if os.path.exists(path):
                path_list.append(path)
            else:
                logger.warning(f"Validation path not found, skipping: {path}")

    if max_samples is not None:
        path_list = path_list[:max_samples]

    if not path_list:
        raise ValueError(f"No valid paths found in CSV: {csv_path}")

    return path_list


def shard_path_list(path_list, rank, world_size):
    local_indices = list(range(rank, len(path_list), world_size))
    local_paths = [path_list[i] for i in local_indices]
    return local_indices, local_paths


def merge_videos_horizontal_ffmpeg(video_paths, output_path):
    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos to concatenate")

    cmd = ["ffmpeg", "-y"]
    for video_path in video_paths:
        cmd.extend(["-i", video_path])

    num_inputs = len(video_paths)
    filter_parts = [f"[{i}:v]setpts=PTS-STARTPTS[v{i}]" for i in range(num_inputs)]
    filter_parts.append("".join(f"[v{i}]" for i in range(num_inputs)) + f"hstack=inputs={num_inputs}[v]")
    filter_complex = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        output_path,
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path



def build_pipeline(args, device, logger):
    cfg = OmegaConf.load(args.config_path)
    load_dtype = torch.bfloat16

    controlnet_path = resolve_controlnet_path(args)
    obj_injector_path = resolve_obj_injector_path(args, controlnet_path)

    logger.info(f"Loading transformer from {args.pretrained_model_path}")
    transformer = PCDController.from_pretrained(
        args.pretrained_model_path,
        subfolder="transformer",
        controlnet_cfg=cfg.controlnet_cfg if cfg else None,
        torch_dtype=load_dtype,
    )
    logger.info(f"Loading ControlNet weights from {controlnet_path}")
    transformer.build_controlnet(model_path=controlnet_path, logger=logger)

    if args.use_object_prompt:
        logger.info(f"Loading object injector weights from {obj_injector_path}")
        transformer.build_object_injector(
            obj_cross_attn_interval=args.obj_cross_attn_interval,
            obj_scale=args.obj_scale,
            traj_mid_dim=args.obj_traj_mid_dim,
            max_entities=args.max_entities,
            num_frames=args.num_frames,
            max_text_tokens=args.max_text_tokens,
            obj_injector_path=obj_injector_path,
            logger=logger,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=load_dtype,
    ).eval()
    image_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model_path, subfolder="image_processor")
    image_encoder = CLIPVisionModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_path,
        subfolder="scheduler",
    )

    pipeline = PCDControllerPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_processor=image_processor,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(device)

    return pipeline, cfg


def save_concat_video(base_path, generated_video_path, concat_output_path, logger):
    sample_name = os.path.basename(base_path)
    original_video_path = os.path.join(base_path, f"{sample_name}.mp4")
    if not os.path.exists(original_video_path):
        logger.warning(f"Original video not found for concat, skipping: {original_video_path}")
        return

    try:
        merge_videos_horizontal_ffmpeg([original_video_path, generated_video_path], concat_output_path)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg or add it to PATH.")


def run_single_sample(args, pipeline, cfg, device, base_path, generated_dir, concat_dir, logger):
    sample_name = os.path.basename(base_path)
    reference_image = os.path.join(base_path, "first_image.png")
    render_path = os.path.join(base_path, "render_output")
    prompt_path = os.path.join(base_path, "full_prompt.json")

    if not os.path.exists(reference_image):
        logger.warning(f"Skip {base_path}: first_image.png not found")
        return False
    if not os.path.exists(render_path):
        logger.warning(f"Skip {base_path}: render_output/ not found")
        return False
    if not os.path.exists(prompt_path):
        logger.warning(f"Skip {base_path}: full_prompt.json not found")
        return False

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    prompt = prompt_data.get("full_prompt", "")

    use_camera_embedding = cfg.get("camera_embedding", args.use_camera_embedding) if cfg else args.use_camera_embedding
    logger.info(f"Loading sample: {sample_name}")
    image, render_video, render_mask, camera_embedding, height, width, camera_3d_preds, object_prompts_list, num_entities = load_validation_dataset(
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

    if args.use_object_prompt and camera_3d_preds is not None:
        camera_3d_preds_input = camera_3d_preds.unsqueeze(0).to(dtype=torch.float32, device=device)
        object_prompts_list_input = [object_prompts_list]
        num_entities_input = torch.tensor([num_entities], device=device)
    else:
        camera_3d_preds_input = None
        object_prompts_list_input = None
        num_entities_input = None

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    logger.info(f"Starting inference for {sample_name}")
    output = pipeline(
        image=image,
        render_video=render_video.to(device),
        render_mask=render_mask.to(device),
        camera_embedding=camera_embedding.to(device) if camera_embedding is not None else None,
        camera_3d_preds=camera_3d_preds_input,
        object_prompts_list=object_prompts_list_input,
        num_entities=num_entities_input,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )

    generated_video = output.frames[0]
    generated_video_path = os.path.join(generated_dir, f"{sample_name}.mp4")
    export_to_video(generated_video, generated_video_path, fps=args.fps)
    logger.info(f"Saved generated video to {generated_video_path}")

    if args.save_concat_video:
        concat_output_path = os.path.join(concat_dir, f"{sample_name}_concat.mp4")
        try:
            save_concat_video(base_path, generated_video_path, concat_output_path, logger)
            if os.path.exists(concat_output_path):
                logger.info(f"Saved concat video to {concat_output_path}")
        except Exception as e:
            logger.warning(f"Failed to create concat video for {sample_name}: {e}")

    del output, generated_video, image, render_video, render_mask, camera_embedding
    if camera_3d_preds_input is not None:
        del camera_3d_preds_input, num_entities_input
    torch.cuda.empty_cache()
    gc.collect()
    return True


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    rank, world_size, local_rank, device = get_runtime_context()
    logger = setup_logger(rank)

    logger.info(f"Using device {device}, world_size={world_size}, local_rank={local_rank}")
    path_list = load_path_list(args.validation_csv_path, args.max_samples, logger)
    local_indices, local_paths = shard_path_list(path_list, rank, world_size)
    logger.info(f"Loaded {len(path_list)} valid paths from {args.validation_csv_path}")
    logger.info(f"Assigned {len(local_paths)} samples with indices: {local_indices}")

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated_videos"
    concat_dir = output_dir / "concat_videos"
    generated_dir.mkdir(parents=True, exist_ok=True)
    if args.save_concat_video:
        concat_dir.mkdir(parents=True, exist_ok=True)

    if not local_paths:
        logger.info("No samples assigned to this rank. Exiting.")
        return

    pipeline, cfg = build_pipeline(args, device, logger)
    logger.info("Model and pipeline initialized once; starting inference loop.")

    success_count = 0
    for base_path in local_paths:
        try:
            success_count += int(run_single_sample(args, pipeline, cfg, device, base_path, str(generated_dir), str(concat_dir), logger))
        except Exception as e:
            logger.exception(f"Inference failed for {base_path}: {e}")

    logger.info(f"Finished {success_count}/{len(local_paths)} assigned samples successfully.")


if __name__ == "__main__":
    main()
