import os
import json

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from src.camera import get_camera_embedding
from src.utils import load_video


def transform_object_trajectories_to_first_camera_frame(camera_3d_preds, c2ws, w2c_first):
    """Transform 3D points from per-frame camera coordinates to the first camera frame coordinate system."""
    if camera_3d_preds is None:
        return None

    if camera_3d_preds.shape[1] != c2ws.shape[0]:
        raise ValueError(
            f"camera_3d_preds frame count {camera_3d_preds.shape[1]} does not match c2ws frame count {c2ws.shape[0]}"
        )

    rotation = torch.matmul(w2c_first[:3, :3].unsqueeze(0), c2ws[:, :3, :3])
    translation = (
        torch.matmul(w2c_first[:3, :3].unsqueeze(0), c2ws[:, :3, 3].unsqueeze(-1)).squeeze(-1)
        + w2c_first[:3, 3].unsqueeze(0)
    )

    camera_3d_preds = torch.as_tensor(camera_3d_preds, dtype=torch.float32)
    transformed = torch.einsum("fij,efpj->efpi", rotation, camera_3d_preds)
    transformed = transformed + translation.unsqueeze(0).unsqueeze(2)
    return transformed.contiguous()



def load_dataset(reference_image, render_path, nframe, max_area, pipe, use_camera_embedding,
                 device, sp_degree=1, logger=None, load_human_info=False, use_object_prompt=False,
                 max_entities=2, normalize_object_to_first_frame=True):
    image = Image.open(reference_image)
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    mod_value = mod_value * sp_degree
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    if logger is not None:
        logger.info(f"Resized image to {height}x{width}")
    image = image.resize((width, height))

    # load conditions
    render_frames = load_video(f"{render_path}/render_with_2d_bbox.mp4")
    render_video = torch.stack([ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
    # replace the first frame with the high-quality reference image
    render_video[0, 0] = ToTensor()(image) * 2 - 1
    render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
    render_mask_frames = load_video(f"{render_path}/render_mask.mp4")
    render_mask = torch.stack([ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
    render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]  # [0,1],[1,f,1,h,w]
    render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1

    # load camera
    # cam_info = json.load(open(f"{render_path}/cam_info.json"))
    # w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
    # intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)
    # intrinsic[0, :] = intrinsic[0, :] / cam_info["width"] * width
    # intrinsic[1, :] = intrinsic[1, :] / cam_info["height"] * height
    # intrinsic = intrinsic[None].repeat(nframe, 1, 1)
    parent_dir = os.path.dirname(render_path)
    # Load and process directly from npz
    npz_files = os.path.join(parent_dir, "spatialtracker2.npz")
    npz_data = dict(np.load(npz_files, allow_pickle=True))
    extrinsics = npz_data["cam_c2w"]
    intrinsic = npz_data["intrinsic"]

    # Convert to tensor and compute w2cs
    K = torch.from_numpy(intrinsic).float()
    c2ws = torch.from_numpy(extrinsics).float()
    w2cs = torch.inverse(c2ws)

    # Use directly, no need to save json
    w2cs = w2cs.to(device)
    intrinsic = K.to(device)

    # Adjust intrinsic based on new width and height
    intrinsic[0, :] = intrinsic[0, :]
    intrinsic[1, :] = intrinsic[1, :]
    intrinsic = intrinsic[None].repeat(nframe, 1, 1)
        
    if use_camera_embedding:
        camera_embedding = get_camera_embedding(intrinsic, w2cs, nframe, height, width, normalize=True)
    else:
        camera_embedding = None

    # Load object trajectory data (if enabled)
    camera_3d_preds = None
    object_prompts_list = None
    num_entities = 0

    if use_object_prompt:
        parent_dir = os.path.dirname(render_path)
        prompt_file = os.path.join(parent_dir, "prompt-didi.json")

        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)

                objects = prompt_data.get('objects', {})
                num_entities = prompt_data.get('object_number', 0)

                # Load camera_3d_preds from NPZ
                camera_3d_list = []
                object_prompts_list = []

                for key in sorted(objects.keys()):
                    sampled_pred_key = f"camera_3d_pred_{key}_sampled"

                    if sampled_pred_key in npz_data:
                        camera_3d = npz_data[sampled_pred_key].copy()
                        camera_3d_list.append(camera_3d)
                    elif len(camera_3d_list) > 0:
                        # Missing entity data, pad with zeros
                        camera_3d_list.append(np.zeros_like(camera_3d_list[0]))
                    else:
                        if logger:
                            logger.warning(f"Missing sampled data for first entity {key}")

                    object_prompts_list.append(objects[key])

                num_entities = len(camera_3d_list)

                # Pad to max_entities
                if len(camera_3d_list) > 0:
                    zero_template = np.zeros_like(camera_3d_list[0])
                    while len(camera_3d_list) < max_entities:
                        camera_3d_list.append(zero_template.copy())
                    while len(object_prompts_list) < max_entities:
                        object_prompts_list.append("")

                    # Stack and convert to tensor: [max_entities, 81, 500, 3]
                    camera_3d_preds = torch.from_numpy(
                        np.stack(camera_3d_list, axis=0)
                    ).float()

                    if normalize_object_to_first_frame:
                        camera_3d_preds = transform_object_trajectories_to_first_camera_frame(
                            camera_3d_preds,
                            c2ws=c2ws,
                            w2c_first=w2cs[0].cpu(),
                        )

            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load object trajectory data: {e}")
                camera_3d_preds = None
                object_prompts_list = None
                num_entities = 0

    if not load_human_info:
        return image, render_video, render_mask, camera_embedding, height, width, camera_3d_preds, object_prompts_list, num_entities
    else:
        smpl_frames = load_video(f"{render_path}/smpl_render.mp4")
        smpl_video = torch.stack([ToTensor()(frame) for frame in smpl_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        smpl_video = F.interpolate(smpl_video, size=(height, width), mode='bicubic')[None]
        smpl_video = torch.clip(smpl_video, -1, 1)  # [-1~1], [1,f,c,h,w]
        hand_frames = load_video(f"{render_path}/hand_render.mp4")
        hand_video = torch.stack([ToTensor()(frame) for frame in hand_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
        hand_video = F.interpolate(hand_video, size=(height, width), mode='bicubic')[None]
        hand_video = torch.clip(hand_video, -1, 1)  # [-1~1], [1,f,c,h,w]

        return image, render_video, render_mask, camera_embedding, smpl_video, hand_video, height, width, camera_3d_preds, object_prompts_list, num_entities
