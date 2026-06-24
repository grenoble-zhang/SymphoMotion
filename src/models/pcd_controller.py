import os
from typing import Any, Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanRotaryPosEmbed
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from huggingface_hub import hf_hub_download
from xfuser.core.distributed import get_sequence_parallel_rank, get_sp_group

from src.models.controlnet import zero_module, WanXControlNet

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SYMPHOMOTION_CHECKPOINT_REPO = os.environ.get("SYMPHOMOTION_CHECKPOINT_REPO", "fateforward/Symphomotion")
CHECKPOINT_FILES = {
    "controlnet.pth": "pretrained_checkpoints/camera_control/controlnet.pth",
    "object_injector.pth": "pretrained_checkpoints/object_control/object_injector.pth",
}


class PerceiverCrossAttention(nn.Module):
    """
    Cross-attention module for injecting object trajectory features into DiT hidden states.
    Q comes from hidden_states (DiT features), KV comes from fused object-trajectory embeddings.
    Follows ConsisID's PerceiverCrossAttention pattern.

    All internal computation is done in float32 to preserve precision of 3D trajectory features.
    """

    def __init__(self, dim: int = 5120, dim_head: int = 128, heads: int = 16, kv_dim: int = 5120):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads  # 128 * 16 = 2048

        self.norm1 = nn.LayerNorm(kv_dim)   # normalizes obj_embeds (KV source)
        self.norm2 = nn.LayerNorm(dim)       # normalizes hidden_states (Q source)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, obj_embeds: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_embeds:    [B, N_obj_tokens, kv_dim]  -- fused object-trajectory embeddings (fp32)
            hidden_states: [B, N_visual, dim]          -- DiT hidden states (may be bf16)
        Returns:
            [B, N_visual, dim] -- same dtype as input hidden_states
        """
        input_dtype = hidden_states.dtype

        # Compute in float32 for precision
        obj_embeds = self.norm1(obj_embeds.float())
        hidden_states_normed = self.norm2(hidden_states.float())

        batch_size, seq_len, _ = hidden_states_normed.shape

        query = self.to_q(hidden_states_normed)
        key, value = self.to_kv(obj_embeds).chunk(2, dim=-1)

        # Reshape for multi-head attention
        query = query.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention (flash attention compatible)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out).to(input_dtype)


class PointNetEncoder(nn.Module):
    """
    Simplified PointNet for per-frame point cloud feature extraction.

    Architecture:
        Input: [B, N, F, P, 3] - 3D point clouds per frame
        Shared MLPs: 3 -> 64 -> 128 -> feat_dim
        Max Pooling: aggregate over points
        Output: [B, N, F, feat_dim] - per-frame features

    Reference: PointNet (Qi et al., CVPR 2017)
    """

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim

        # Shared MLPs (implemented as Conv1d for efficiency)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feat_dim, 1)

        # Batch normalization with small momentum for batch_size=1 stability
        # track_running_stats=False to avoid in-place updates during training
        self.bn1 = nn.BatchNorm1d(64, momentum=0.01, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.01, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(feat_dim, momentum=0.01, track_running_stats=False)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F, P, 3] - point clouds
        Returns:
            [B, N, F, feat_dim] - per-frame features
        """
        B, N, F, P, _ = x.shape

        # Reshape to [B*N*F, 3, P] for Conv1d processing
        x = x.reshape(B * N * F, P, 3).transpose(1, 2).contiguous()  # [B*N*F, 3, P]

        # Shared MLPs with BatchNorm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))  # [B*N*F, 64, P]
        x = self.relu(self.bn2(self.conv2(x)))  # [B*N*F, 128, P]
        x = self.relu(self.bn3(self.conv3(x)))  # [B*N*F, feat_dim, P]

        # Max pooling over points (symmetric aggregation function)
        x = torch.max(x, dim=2)[0]  # [B*N*F, feat_dim]

        # Reshape back to [B, N, F, feat_dim]
        x = x.reshape(B, N, F, self.feat_dim)
        return x


class ObjectTrajectoryEncoder(nn.Module):
    """
    Encodes per-entity 3D trajectory points and text prompts into fused embeddings.
    Following 3DTrajMaster architecture with pairwise text-trajectory fusion.
    Uses PointNet for point cloud encoding instead of simple mean pooling.
    All computation is done in float32 to preserve 3D trajectory precision.

    Architecture (3DTrajMaster style with PointNet):
        1. Trajectory: PointNet(3D points) -> [B, max_ent, num_frames, traj_mid_dim]
           Then Linear(traj_mid_dim, inner_dim) per frame
           Output: [B, max_ent, num_frames, inner_dim]
        2. Text: pre-encoded T5 embeddings [B, max_ent, seq_len, text_dim] -> Linear(text_dim, inner_dim)
           Apply attention mask, truncate to max_text_tokens
           Output: [B, max_ent, max_text_tokens, inner_dim]
        3. Pairwise Fusion: text.unsqueeze(-2) + traj.unsqueeze(-3)
           Output: [B, max_ent, max_text_tokens, num_frames, inner_dim]
        4. Mask unused entities based on num_entities
        5. Flatten: [B, max_ent * max_text_tokens * num_frames, inner_dim]
    """

    def __init__(self, inner_dim: int = 5120, text_dim: int = 4096, traj_mid_dim: int = 512,
                 max_entities: int = 2, num_frames: int = 81, max_text_tokens: int = 50):
        super().__init__()
        self.inner_dim = inner_dim
        self.max_entities = max_entities
        self.num_frames = num_frames
        self.max_text_tokens = max_text_tokens

        # Trajectory encoding: PointNet for point cloud feature extraction
        self.pointnet = PointNetEncoder(feat_dim=traj_mid_dim)
        self.traj_frame_proj = nn.Linear(traj_mid_dim, inner_dim)

        # Text encoding: project T5 embeddings to inner_dim
        self.text_proj = nn.Linear(text_dim, inner_dim)

    def forward(self, camera_3d_preds: torch.Tensor, obj_text_embeds: torch.Tensor,
                obj_text_masks: torch.Tensor, num_entities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_3d_preds: [B, max_ent, num_frames, num_points, 3] float32
            obj_text_embeds: [B, max_ent, seq_len, text_dim] (full T5 sequence embeddings)
            obj_text_masks:  [B, max_ent, seq_len] (attention mask, 1=valid, 0=padding)
            num_entities:    [B] (actual number of entities per sample)
        Returns:
            fused_embeds: [B, max_ent * max_text_tokens * num_frames, inner_dim] float32
        """
        B = camera_3d_preds.shape[0]
        E = camera_3d_preds.shape[1]  # max_entities
        F = camera_3d_preds.shape[2]  # num_frames

        # Ensure float32 computation for trajectory precision (preserve device)
        camera_3d_preds = camera_3d_preds.to(dtype=torch.float32)
        obj_text_embeds = obj_text_embeds.to(dtype=torch.float32)
        obj_text_masks = obj_text_masks.to(dtype=torch.float32)

        # 1. Trajectory encoding with PointNet
        traj_feat = self.pointnet(camera_3d_preds)       # [B, E, F, traj_mid_dim]
        traj_feat = self.traj_frame_proj(traj_feat)      # [B, E, F, inner_dim]

        # 2. Text encoding
        text_feat = self.text_proj(obj_text_embeds)          # [B, E, seq_len, inner_dim]

        # Apply attention mask (zero out padding tokens)
        text_feat = text_feat * obj_text_masks.unsqueeze(-1)  # [B, E, seq_len, inner_dim]

        # Truncate to max_text_tokens (like 3DTrajMaster truncates to 50)
        text_feat = text_feat[:, :, :self.max_text_tokens, :]  # [B, E, max_text_tokens, inner_dim]

        # 3. Pairwise addition (3DTrajMaster style)
        # traj_feat: [B, E, F, inner_dim] -> unsqueeze(-3) -> [B, E, 1, F, inner_dim]
        # text_feat: [B, E, T, inner_dim] -> unsqueeze(-2) -> [B, E, T, 1, inner_dim]
        # Broadcasting: [B, E, T, F, inner_dim]
        fused = traj_feat.unsqueeze(-3) + text_feat.unsqueeze(-2)  # [B, E, max_text_tokens, F, inner_dim]

        # Free intermediate tensors
        del traj_feat, text_feat

        # 4. Mask unused entities (zero out entities beyond num_entities)
        ent_idx = torch.arange(E, device=fused.device)
        mask = (ent_idx.unsqueeze(0) < num_entities.unsqueeze(1)).float()  # [B, E]
        # Broadcast mask to [B, E, T, F, inner_dim]
        fused = fused * mask.view(B, E, 1, 1, 1)
        del mask, ent_idx

        # 5. Flatten: [B, E, T, F, D] -> [B, E*T*F, D]
        fused = fused.reshape(B, -1, self.inner_dim)
        return fused


class MaskCamEmbed(nn.Module):
    def __init__(self, controlnet_cfg) -> None:
        super().__init__()

        # padding bug fixed
        if controlnet_cfg.get("interp", False):
            self.mask_padding = [0, 0, 0, 0, 3, 3]  # left, right, top, bottom, front, back; I2V-interp, first and last frames
        else:
            self.mask_padding = [0, 0, 0, 0, 3, 0]  # left, right, top, bottom, front, back; I2V
        add_channels = controlnet_cfg.get("add_channels", 1)
        mid_channels = controlnet_cfg.get("mid_channels", 64)
        self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8)),
                                       nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        self.mask_zero_proj = zero_module(nn.Conv3d(mid_channels, controlnet_cfg.conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)))

    def forward(self, add_inputs: torch.Tensor):
        # render_mask.shape [b,c,f,h,w]
        warp_add_pad = F.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        add_embeds = self.mask_proj(warp_add_pad)  # [B,C,F,H,W]
        add_embeds = self.mask_zero_proj(add_embeds)
        add_embeds = einops.rearrange(add_embeds, "b c f h w -> b (f h w) c")

        return add_embeds


class PCDController(WanTransformer3DModel):
    r"""
    A Transformer model for video-like data used in the Wan model.
    """

    def __init__(
            self,
            patch_size: Tuple[int] = (1, 2, 2),
            num_attention_heads: int = 40,
            attention_head_dim: int = 128,
            in_channels: int = 16,
            out_channels: int = 16,
            text_dim: int = 4096,
            freq_dim: int = 256,
            ffn_dim: int = 13824,
            num_layers: int = 40,
            cross_attn_norm: bool = True,
            qk_norm: Optional[str] = "rms_norm_across_heads",
            eps: float = 1e-6,
            image_dim: Optional[int] = None,
            added_kv_proj_dim: Optional[int] = None,
            rope_max_seq_len: int = 1024,
            controlnet_cfg=None
    ) -> None:
        super().__init__(patch_size=patch_size,
                         num_attention_heads=num_attention_heads,
                         attention_head_dim=attention_head_dim,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         text_dim=text_dim,
                         freq_dim=freq_dim,
                         ffn_dim=ffn_dim,
                         num_layers=num_layers,
                         cross_attn_norm=cross_attn_norm,
                         qk_norm=qk_norm,
                         eps=eps,
                         image_dim=image_dim,
                         added_kv_proj_dim=added_kv_proj_dim,
                         rope_max_seq_len=rope_max_seq_len)

        self.controlnet_cfg = controlnet_cfg
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.rope_max_seq_len = rope_max_seq_len
        self.sp_size = 1

    def build_controlnet(self, model_path, logger=None):
        # controlnet
        self.controlnet_patch_embedding = nn.Conv3d(
            self.in_channels, self.controlnet_cfg.conv_out_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.controlnet_mask_embedding = MaskCamEmbed(self.controlnet_cfg)
        self.controlnet = WanXControlNet(self.controlnet_cfg)
        self.controlnet_rope = WanRotaryPosEmbed(self.controlnet_cfg.dim // self.controlnet_cfg.num_heads,
                                                 self.patch_size, self.rope_max_seq_len)

        if not os.path.exists(model_path):  # download weights from Hugging Face
            filename = CHECKPOINT_FILES.get(os.path.basename(model_path), model_path.replace("\\", "/"))
            model_path = hf_hub_download(repo_id=SYMPHOMOTION_CHECKPOINT_REPO, filename=filename, repo_type="model")
        state_dict = torch.load(model_path, map_location="cpu")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        # print("Missing keys:", missing_keys)
        if logger is not None:
            logger.info(f"Unexpected keys: {unexpected_keys}")
        else:
            print("Unexpected keys:", unexpected_keys)

    def build_object_injector(self, obj_cross_attn_interval: int = 2, obj_scale: float = 1.0,
                              traj_mid_dim: int = 512, max_entities: int = 2, num_frames: int = 81,
                              max_text_tokens: int = 50, obj_injector_path: str = None, logger=None):
        """Build object trajectory injection modules with PointNet encoder (called after build_controlnet)."""
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim  # 40 * 128 = 5120
        text_dim = self.config.text_dim  # 4096
        num_layers = self.config.num_layers  # 40

        self.obj_cross_attn_interval = obj_cross_attn_interval
        self.obj_scale = obj_scale
        # Inject every obj_cross_attn_interval layers in first 20 layers
        # e.g., interval=2 -> inject at layers 0,2,4,6,8,10,12,14,16,18 (10 modules)
        self.num_obj_cross_attn = 20 // obj_cross_attn_interval

        # Trajectory + text encoder with PointNet (float32)
        self.obj_traj_encoder = ObjectTrajectoryEncoder(
            inner_dim=inner_dim,
            text_dim=text_dim,
            traj_mid_dim=traj_mid_dim,
            max_entities=max_entities,
            num_frames=num_frames,
            max_text_tokens=max_text_tokens,
        )
        # Ensure obj_traj_encoder is in float32 (including BatchNorm buffers) and on correct device
        self.obj_traj_encoder.to(device=self.device, dtype=torch.float32)

        # PerceiverCrossAttention modules (one per injection point, float32)
        self.obj_perceiver_cross_attention = nn.ModuleList([
            PerceiverCrossAttention(
                dim=inner_dim,
                dim_head=self.config.attention_head_dim,  # 128
                heads=16,
                kv_dim=inner_dim,
            )
            for _ in range(self.num_obj_cross_attn)
        ])
        # Ensure perceiver modules are in float32
        for ca in self.obj_perceiver_cross_attention:
            ca.to(torch.float32)

        # Zero-initialize to_out so injection starts from zero (pretrained behavior preserved)
        for ca in self.obj_perceiver_cross_attention:
            nn.init.zeros_(ca.to_out.weight)

        # Optionally load pretrained object injector weights
        if obj_injector_path:
            if not os.path.exists(obj_injector_path):
                filename = CHECKPOINT_FILES.get(os.path.basename(obj_injector_path),
                                                obj_injector_path.replace("\\", "/"))
                obj_injector_path = hf_hub_download(repo_id=SYMPHOMOTION_CHECKPOINT_REPO,
                                                    filename=filename,
                                                    repo_type="model")
            self.load_object_injector(obj_injector_path, logger=logger)

        # Log parameter count
        total_params = sum(p.numel() for p in self.obj_traj_encoder.parameters())
        total_params += sum(p.numel() for p in self.obj_perceiver_cross_attention.parameters())
        if logger is not None:
            logger.info(f"Object injector built with PointNet: {total_params:,} parameters, "
                        f"interval={obj_cross_attn_interval}, num_modules={self.num_obj_cross_attn}")
        else:
            print(f"Object injector built with PointNet: {total_params:,} parameters, "
                  f"interval={obj_cross_attn_interval}, num_modules={self.num_obj_cross_attn}")

    def save_object_injector(self, path: str):
        """Save object injector modules to a file."""
        save_dict = {
            "obj_traj_encoder": self.obj_traj_encoder.state_dict(),
            "obj_perceiver_cross_attention": [
                ca.state_dict() for ca in self.obj_perceiver_cross_attention
            ],
            "obj_cross_attn_interval": self.obj_cross_attn_interval,
            "obj_scale": self.obj_scale,
        }
        torch.save(save_dict, path)

    def load_object_injector(self, path_or_state_dict, logger=None):
        """Load object injector modules from a file path or state dict."""
        if isinstance(path_or_state_dict, str):
            path_or_state_dict = torch.load(path_or_state_dict, map_location="cpu")

        self.obj_traj_encoder.load_state_dict(path_or_state_dict["obj_traj_encoder"])
        for ca, sd in zip(self.obj_perceiver_cross_attention,
                          path_or_state_dict["obj_perceiver_cross_attention"]):
            ca.load_state_dict(sd)

        if logger:
            logger.info("Loaded object injector weights successfully")
        else:
            print("Loaded object injector weights successfully")

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: Optional[torch.Tensor] = None,
            render_latent=None,
            render_mask=None,
            camera_embedding=None,
            obj_embeds=None,
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param render_latent: [b,c,f,h,w]
        :param render_mask: [b,1,f,h,w]
        :param camera_embedding: [b,6,f,h,w]
        :param obj_embeds: [b, max_ent*num_frames, inner_dim] fused object-trajectory embeddings (fp32)
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        ### process controlnet inputs ###
        render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)
        controlnet_rotary_emb = self.controlnet_rope(render_latent)
        controlnet_inputs = self.controlnet_patch_embedding(render_latent)
        controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)

        # additional inputs (mask, camera embedding)
        if camera_embedding is not None:
            add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
        else:
            add_inputs = render_mask
        add_inputs = self.controlnet_mask_embedding(add_inputs)
        controlnet_inputs = controlnet_inputs + add_inputs
        ### process controlnet inputs over ###

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        ### controlnet encoding ###
        if self.sp_size > 1:
            assert controlnet_inputs.shape[1] % self.sp_size == 0
            controlnet_inputs = torch.chunk(controlnet_inputs, self.sp_size, dim=1)[get_sequence_parallel_rank()]
            controlnet_rotary_emb = torch.chunk(controlnet_rotary_emb, self.sp_size, dim=2)[get_sequence_parallel_rank()]

        with torch.autocast("cuda", dtype=self.dtype, enabled=True):
            controlnet_states = self.controlnet(hidden_states=controlnet_inputs,
                                                temb=temb,
                                                rotary_emb=controlnet_rotary_emb)
        ### controlnet encoding over ###

        ### sp
        if self.sp_size > 1:
            assert hidden_states.shape[1] % self.sp_size == 0
            hidden_states = torch.chunk(hidden_states, self.sp_size, dim=1)[get_sequence_parallel_rank()]
            rotary_emb = torch.chunk(rotary_emb, self.sp_size, dim=2)[get_sequence_parallel_rank()]

        # 4. Transformer blocks
        has_obj_injector = obj_embeds is not None and hasattr(self, 'obj_perceiver_cross_attention')

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                # adding camera control features (frozen in Stage 2)
                if i < len(controlnet_states):
                    hidden_states = hidden_states + controlnet_states[i]
                # adding object trajectory features (new in Stage 2) - inject every interval layers in first 20 layers
                if has_obj_injector and i < 20 and i % self.obj_cross_attn_interval == 0:
                    module_idx = i // self.obj_cross_attn_interval
                    hidden_states = hidden_states + self.obj_scale * \
                        self.obj_perceiver_cross_attention[module_idx](obj_embeds, hidden_states)
        else:
            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                # adding camera control features (frozen in Stage 2)
                if i < len(controlnet_states):
                    hidden_states = hidden_states + controlnet_states[i]
                # adding object trajectory features (new in Stage 2) - inject every interval layers in first 20 layers
                if has_obj_injector and i < 20 and i % self.obj_cross_attn_interval == 0:
                    module_idx = i // self.obj_cross_attn_interval
                    hidden_states = hidden_states + self.obj_scale * \
                        self.obj_perceiver_cross_attention[module_idx](obj_embeds, hidden_states)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if self.sp_size > 1:
            hidden_states = get_sp_group().all_gather(hidden_states, dim=1)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
