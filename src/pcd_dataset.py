"""
PCDDataset for training - Dataset for PCD (Point Cloud Diffusion) Controller training.
Preprocesses and returns all data required for training.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from torchvision.transforms import ToTensor
# When running this file directly, add project root to sys.path for importing src
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if (_project_root / "src").is_dir() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from src.camera import get_camera_embedding
from src.utils import load_video


def load_paths_from_csv(csv_path):
    """Read path list from CSV file"""
    paths = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Check if 'path' column exists
            if 'path' not in reader.fieldnames:
                raise ValueError(f"'path' column not found in CSV file. Available columns: {reader.fieldnames}")

            for row in reader:
                path = row['path'].strip()
                if path and os.path.exists(path):
                    paths.append(path)
                elif path:
                    print(f"Warning: Path does not exist, skipping: {path}")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        raise

    return paths


def load_prompt_data(prompt_path, logger=None):
    """Load prompt data, return detailed objects information and actual object count (no duplication for single object)"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

    # Get objects information, maintain original key-value mapping
    objects = prompt_data.get('objects', {})

    # Return actual object count, no duplication
    object_number = len(objects)
    return {
        'objects': objects,
        'object_number': object_number,
    }


def _apply_tracker_sampling(track2d, camera_3d, tracker_list):
    """Apply tracker's filtering_stats.sampled_indices per frame, return (track2d, camera_3d)."""
    if not isinstance(tracker_list, list) or len(tracker_list) == 0:
        return track2d, camera_3d
    sampled_t, sampled_c = [], []
    for t, frame_data in enumerate(tracker_list):
        if t >= track2d.shape[0]:
            break
        filt = frame_data.get("filtering_stats", {}) if isinstance(frame_data, dict) else {}
        idx = filt.get("sampled_indices", None) if isinstance(filt, dict) else None
        if idx and isinstance(idx, (list, np.ndarray)) and len(idx) > 0:
            idx = np.asarray(idx, dtype=np.int64)
            sampled_t.append(track2d[t][idx])
            sampled_c.append(camera_3d[t][idx])
        else:
            sampled_t.append(track2d[t])
            sampled_c.append(camera_3d[t])
    if len(sampled_t) == track2d.shape[0]:
        return np.stack(sampled_t, axis=0), np.stack(sampled_c, axis=0)
    return track2d, camera_3d



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


class PCDDataset(Dataset):
    """
    Dataset for loading PCD Controller training data with video, render, camera info, etc.
    Dataset for camera control training, returns original images, videos, renders, and camera information.
    """
    
    def __init__(
        self,
        csv_path: str,
        num_frames: int = 81,
        max_area: int = 480 * 832,
        use_camera_embedding: bool = True,
        use_object_prompt: bool = False,
        max_entities: int = 2,
        device: str = "cpu",
        normalize_object_to_first_frame: bool = True,
    ):
        """
        Args:
            csv_path: Path to CSV file containing sample paths in 'path' column
            num_frames: Number of frames per video
            max_area: Maximum pixel area for resizing (height * width)
            use_camera_embedding: Whether to compute camera embeddings
            use_object_prompt: Whether to use object-specific prompts
            max_entities: Maximum number of object entities (zero-padded for fewer)
            device: Device to load data to ('cpu' or 'cuda')
            normalize_object_to_first_frame: Whether to transform object trajectories from per-frame camera coordinates into the first-frame camera coordinate system
        """
        self.csv_path = csv_path
        self.num_frames = num_frames
        self.max_area = max_area
        self.use_camera_embedding = use_camera_embedding
        self.use_object_prompt = use_object_prompt
        self.max_entities = max_entities
        self.device = device
        self.normalize_object_to_first_frame = normalize_object_to_first_frame
        self.width = 832
        self.height = 480
        # Read CSV file and get valid paths
        self.sample_paths = load_paths_from_csv(self.csv_path)
        print(f"Loaded {len(self.sample_paths)} valid paths from CSV")
        
        # Find all valid samples
        self.samples = []
        self._collect_samples()
    
    def _collect_samples(self):
        """Collect all valid samples from CSV file"""
        # Process each sample from the loaded path list
        for sample_path in self.sample_paths:
            sample_dir = Path(sample_path)

            # Check for first_image.png
            ref_image = sample_dir / "first_image.png"

            # render_dir is render_output under the path
            render_dir = sample_dir / "render_output"

            # prompt_file is prompt-didi.json under the path
            prompt_file = sample_dir / "prompt-didi.json"

            # full_prompt_file is full_prompt.json under the path
            full_prompt_file = sample_dir / "full_prompt.json"

            # spatialtracker2.npz is in sample_dir directory
            npz_file = sample_dir / "spatialtracker2.npz"

            # Original video file, named as <sample_dir_name>.mp4
            gt_video_file = sample_dir / f"{sample_dir.name}.mp4"

            self.samples.append({
                'sample_id': str(sample_dir),
                'base_name': sample_dir.name,
                'reference_image': ref_image,
                'render_dir': render_dir,
                'prompt_file': prompt_file,
                'full_prompt_file': full_prompt_file,
                'npz_file': npz_file,
                'gt_video': gt_video_file,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and preprocess a single sample
        Returns all necessary data for training including latents and embeddings
        """
        max_retries = 30
        for _retry in range(max_retries):
            try:
                return self._load_sample(idx)
            except Exception as e:
                print(f"Warning: Failed to load sample {idx}: {e}. "
                      f"Retry {_retry + 1}/{max_retries}.")
                time.sleep(1)  # 等待1秒，给 NFS 恢复的时间
                idx = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"Failed to load a valid sample after {max_retries} retries")

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load reference image (固定尺寸 832x480，无需 resize)
        image = Image.open(sample['reference_image']).convert('RGB')
        image = ToTensor()(image)
        # 使用固定的尺寸，不需要计算和 resize
        # aspect_ratio = image.height / image.width
        # mod_value = 16
        # height = round(np.sqrt(self.max_area * aspect_ratio)) // mod_value * mod_value
        # width = round(np.sqrt(self.max_area / aspect_ratio)) // mod_value * mod_value
        # image = image.resize((width, height))
        
        # Load render video (使用 render_with_2d_bbox.mp4, 已经是 832x480，无需 resize)
        render_frames = load_video(sample['render_dir'] / "render_with_2d_bbox.mp4")
        render_video = torch.stack([ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0
        # render_video = F.interpolate(render_video, size=(self.height, self.width), mode='bicubic')[None]
        render_video = render_video[None]  # 只添加 batch 维度
        # Replace first frame with high-quality reference
        # render_video[0, 0] = ToTensor()(image) * 2 - 1
        render_video = torch.clip(render_video, -1, 1)
        render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")

        # Load ground-truth video (原始视频，用于 flow matching 的去噪目标)
        gt_frames = load_video(str(sample['gt_video']))
        gt_video = torch.stack([ToTensor()(frame) for frame in gt_frames], dim=0) * 2 - 1.0
        gt_video = gt_video[None]
        gt_video = torch.clip(gt_video, -1, 1)
        gt_video = einops.rearrange(gt_video, "b f c h w -> b c f h w")
        
        # Load render mask (已经是 832x480，无需 resize)
        render_mask_frames = load_video(sample['render_dir'] / "render_mask.mp4")
        render_mask = torch.stack([ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]
        # render_mask = F.interpolate(render_mask, size=(self.height, self.width), mode='nearest')[None]
        render_mask = render_mask[None]  # 只添加 batch 维度
        render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
        render_mask[render_mask < 0.5] = 0
        render_mask[render_mask >= 0.5] = 1
        
        # Load camera info from npz file (spatialtracker2.npz 在 sample_dir 目录下)
        npz_data = dict(np.load(sample['npz_file'], allow_pickle=True))
        extrinsics = npz_data["cam_c2w"]
        intrinsic_raw = npz_data["intrinsic"]
        
        # use_object_prompt 为 False 时跳过：仅从 npz 取原始数据，不做 copy 与 2D 采样
        camera_3d_preds = None      # [max_entities, 81, 500, 3] or None
        object_prompts_list = None  # list[str], len=max_entities
        num_entities = 0

        # 物体轨迹与 prompt：仅 use_object_prompt 时从 npz 读取并做 2D 包围框采样
        if self.use_object_prompt:
            # 加载 prompt 数据
            prompt_file = sample['prompt_file']
            prompt_data = load_prompt_data(prompt_file)
            objects = prompt_data.get('objects', {})
            num_entities = prompt_data.get('object_number', 0)

            # 加载 bounding box 采样信息
            bounding_data = {}
            bounding_boxes_2d_json_file = sample['render_dir'] / "bounding_boxes_2d.json"
            try:
                with open(bounding_boxes_2d_json_file, 'r', encoding='utf-8') as f:
                    bounding_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load bounding_boxes_2d.json from {bounding_boxes_2d_json_file}: {e}")

            # 按实际物体数量加载轨迹数据，不做复制
            camera_3d_list = []
            object_prompts_list = []
            for key in sorted(objects.keys()):  # '0', '1', '2' ...
                # 只加载下采样后的数据
                sampled_pred_key = f"camera_3d_pred_{key}_sampled"

                if sampled_pred_key in npz_data:
                    # 使用预处理好的下采样数据
                    camera_3d = npz_data[sampled_pred_key].copy()
                    camera_3d_list.append(camera_3d)
                elif len(camera_3d_list) > 0:
                    # npz 中缺失该 entity 的数据，用零填充
                    camera_3d_list.append(np.zeros_like(camera_3d_list[0]))
                else:
                    # 第一个entity就缺失，无法创建零模板，跳过
                    print(f"Warning: Missing sampled data for first entity {key} in {sample['sample_id']}")

                object_prompts_list.append(objects[key])

            num_entities = len(camera_3d_list)

            # Pad to max_entities with zeros
            if len(camera_3d_list) > 0:
                zero_template = np.zeros_like(camera_3d_list[0])
                while len(camera_3d_list) < self.max_entities:
                    camera_3d_list.append(zero_template.copy())
                while len(object_prompts_list) < self.max_entities:
                    object_prompts_list.append("")

                # Stack: [max_entities, 81, 500, 3]
                camera_3d_preds = np.stack(camera_3d_list, axis=0)

            # Stage 2 object-only training requires at least one valid entity.
            # If no valid trajectory is available, force __getitem__ retry.
            if num_entities == 0 or camera_3d_preds is None:
                raise ValueError(
                    f"No valid object trajectories for sample: {sample['sample_id']}"
                )


        # 转换为tensor并计算w2cs
        K = torch.from_numpy(intrinsic_raw).float()
        c2ws = torch.from_numpy(extrinsics).float()
        w2cs = torch.inverse(c2ws)

        if self.normalize_object_to_first_frame and camera_3d_preds is not None:
            camera_3d_preds = transform_object_trajectories_to_first_camera_frame(
                camera_3d_preds,
                c2ws=c2ws,
                w2c_first=w2cs[0],
            ).numpy()

        # 使用原始intrinsic，不需要额外的缩放
        # 可以简化为：
        intrinsic = K[None].repeat(self.num_frames, 1, 1)
        
        # Compute camera embedding
        camera_embedding = None
        if self.use_camera_embedding:
            camera_embedding = get_camera_embedding(
                intrinsic, w2cs, self.num_frames, self.height, self.width, normalize=True
            )

        # prompt 直接从 full_prompt.json 读取 "full_prompt" 字段
        full_prompt_file = sample['full_prompt_file']
        with open(full_prompt_file, 'r', encoding='utf-8') as f:
            full_prompt_json = json.load(f)
        prompt = full_prompt_json.get('full_prompt', '')

        # Return data dict
        # batch 后形状参考：render_video [B,3,81,480,832], render_mask [B,1,81,480,832],
        # camera_3d_preds [B, max_entities, 81, 500, 3], camera_embedding [B,6,81,480,832]
        return {
            'sample_id': sample['sample_id'],                    # list len=B
            'base_name': sample['base_name'],                     # list len=B
            'image': image,                                       # tensor [3,H,W], list len=B
            'render_video': render_video.squeeze(0),              # [c,f,h,w] -> batch [B,3,81,480,832]
            'gt_video': gt_video.squeeze(0),                      # [c,f,h,w] -> batch [B,3,81,480,832]
            'render_mask': render_mask.squeeze(0),                # [c,f,h,w] -> batch [B,1,81,480,832]
            'camera_embedding': camera_embedding.squeeze(0),      # [f,dim] or None -> batch [B,6,81,480,832]
            'height': self.height,                                # int -> batch shape [B] dtype=int64
            'width': self.width,                                  # int -> batch shape [B] dtype=int64
            'prompt': prompt,                                     # list len=B
            'object_prompts_list': object_prompts_list,           # list[str] len=max_entities or None, list[list] len=B
            'num_entities': num_entities,                          # int -> batch [B] dtype=int64
            'camera_3d_preds': camera_3d_preds,                   # [max_entities, 81, 500, 3] or None -> batch [B, max_entities, 81, 500, 3]
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle batching of samples
    """
    # 将 numpy 转为 tensor 并 stack
    def _to_tensor(x):
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)

    batch_dict = {
        'sample_id': [item['sample_id'] for item in batch],
        'base_name': [item['base_name'] for item in batch],
        'image': [item['image'] for item in batch],  # List of tensors
        'render_video': torch.stack([item['render_video'] for item in batch]),
        'gt_video': torch.stack([item['gt_video'] for item in batch]),
        'render_mask': torch.stack([item['render_mask'] for item in batch]),
        'height': torch.tensor([item['height'] for item in batch], dtype=torch.long),
        'width': torch.tensor([item['width'] for item in batch], dtype=torch.long),
        'prompt': [item['prompt'] for item in batch],
    }

    # object_prompts_list: list[list[str]] — 每个样本有 max_entities 个字符串（含空串 padding）
    batch_dict['object_prompts_list'] = [item['object_prompts_list'] for item in batch]

    # num_entities: [B] — 每个样本的实际物体数量
    batch_dict['num_entities'] = torch.tensor(
        [item['num_entities'] for item in batch], dtype=torch.long
    )

    # camera_3d_preds: [B, max_entities, 81, 500, 3] — 已 pad 到 max_entities，直接 stack
    vals = [item['camera_3d_preds'] for item in batch]
    if all(v is not None for v in vals):
        batch_dict['camera_3d_preds'] = torch.stack([_to_tensor(v) for v in vals])
    else:
        batch_dict['camera_3d_preds'] = None

    # Handle optional camera embedding
    if batch[0]['camera_embedding'] is not None:
        batch_dict['camera_embedding'] = torch.stack([item['camera_embedding'] for item in batch])
    else:
        batch_dict['camera_embedding'] = None

    return batch_dict


def main():
    """Quick dataset sanity check."""
    parser = argparse.ArgumentParser(description="Validate a SymphoMotion CSV manifest.")
    parser.add_argument("--csv_path", required=True, help="CSV manifest with a 'path' column.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--use_camera_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_object_prompt", action="store_true", default=False)
    parser.add_argument("--max_entities", type=int, default=2)
    parser.add_argument("--normalize_object_to_first_frame", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"CSV not found: {args.csv_path}")
        sys.exit(1)

    print("Creating PCDDataset...")
    dataset = PCDDataset(
        csv_path=args.csv_path,
        num_frames=args.num_frames,
        use_camera_embedding=args.use_camera_embedding,
        use_object_prompt=args.use_object_prompt,
        max_entities=args.max_entities,
        normalize_object_to_first_frame=args.normalize_object_to_first_frame,
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty.")
        sys.exit(1)

    print(f"\nCreating DataLoader(batch_size={args.batch_size}, collate_fn=custom_collate_fn)...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )

    print(f"\nReading the first batch (batch_size={args.batch_size})...")
    batch = next(iter(dataloader))
    for k, v in batch.items():
        if k == "num_entities":
            print(f"  {k}: {v.tolist()}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], list):
                print(f"  {k}: list[list] len={len(v)}, inner_len={len(v[0])}, sample={v[0]}")
            else:
                print(f"  {k}: list len={len(v)}")
        else:
            print(f"  {k}: {type(v).__name__}")

    print(f"\nIterating over 2 more batches...")
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        b = batch["render_video"].shape[0]
        ne = batch["num_entities"].tolist()
        print(f"  batch {i + 1}: batch_size={b}, num_entities={ne}")

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
