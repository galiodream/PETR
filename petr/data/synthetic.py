"""
Synthetic Multi-View Dataset for PETR.

Generates synthetic multi-view images with camera parameters for
3D object detection training.

This dataset simulates:
- Multi-view camera setup with realistic intrinsics/extrinsics
- 3D bounding boxes with proper 2D projections
- Camera parameter tensors for 3D position embedding
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from petr.utils.seed import seed_worker


@dataclass
class SyntheticConfig:
    """Configuration for synthetic dataset."""
    image_size: Tuple[int, int]
    num_views: int
    num_classes: int
    num_queries: int
    seed: int
    use_3d_bbox: bool = False  # Whether to use 3D bounding boxes


def generate_camera_intrinsics(
    batch_size: int,
    num_views: int,
    image_size: Tuple[int, int],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate camera intrinsics matrix K.

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    For synthetic data, we use a default FOV of 60 degrees.
    """
    h, w = image_size
    fx = fy = w / (2 * torch.tan(torch.tensor(torch.pi / 6)))  # FOV = 60 deg
    cx, cy = w / 2, h / 2

    K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    K = K.expand(batch_size, num_views, -1, -1).clone()
    K[:, :, 0, 0] = fx
    K[:, :, 1, 1] = fy
    K[:, :, 0, 2] = cx
    K[:, :, 1, 2] = cy

    return K


def generate_camera_extrinsics(
    batch_size: int,
    num_views: int,
    radius: float = 10.0,
    height: float = 2.0,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate camera extrinsics matrix (world to camera transformation).

    For multi-view setup, cameras are placed in a circle around the scene.

    E = [[R | t],
         [0 | 1]]

    Where R is rotation and t is translation.
    """
    # Camera positions in a circle
    angles = torch.linspace(0, 2 * torch.pi, num_views + 1, device=device, dtype=dtype)[:-1]

    # Camera positions (world coordinates)
    cam_x = radius * torch.cos(angles)
    cam_y = torch.full_like(angles, height)
    cam_z = radius * torch.sin(angles)
    cam_positions = torch.stack([cam_x, cam_y, cam_z], dim=-1)  # (V, 3)

    # Rotation matrix: camera looks at origin
    # For each camera, we need to compute R such that the camera looks at (0, 0, 0)
    extrinsics = torch.zeros((batch_size, num_views, 4, 4), device=device, dtype=dtype)
    extrinsics[:, :, 3, 3] = 1

    for v in range(num_views):
        cam_pos = cam_positions[v]

        # Forward vector (camera z-axis points away from scene, but we want it to point at scene)
        forward = -cam_pos / torch.norm(cam_pos)

        # Right vector (camera x-axis)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        right = torch.cross(world_up, forward, dim=-1)
        right = right / torch.norm(right)

        # Up vector (camera y-axis)
        up = torch.cross(forward, right, dim=-1)

        # Rotation matrix (world to camera)
        R = torch.stack([right, up, forward], dim=-1).T  # (3, 3)

        # Translation
        t = -R @ cam_pos

        extrinsics[:, v, :3, :3] = R
        extrinsics[:, v, :3, 3] = t

    return extrinsics


class SyntheticMultiViewDataset(Dataset):
    """
    Synthetic Multi-View Dataset for PETR.

    Generates random multi-view images with corresponding bounding boxes
    and camera parameters.

    For training PETR with 3D position embeddings, this dataset provides:
    - images: (V, 3, H, W) multi-view images
    - boxes: (N, D) bounding boxes (D=4 for 2D, D=8 for 3D)
    - labels: (N,) class labels
    - cam_intrinsics: (V, 3, 3) camera intrinsics
    - cam_extrinsics: (V, 4, 4) camera extrinsics
    """

    def __init__(
        self,
        num_samples: int,
        cfg: SyntheticConfig,
    ) -> None:
        self.num_samples = num_samples
        self.cfg = cfg

        # Pre-generate camera parameters
        self.cam_intrinsics = generate_camera_intrinsics(
            1, cfg.num_views, cfg.image_size
        ).squeeze(0)  # (V, 3, 3)

        self.cam_extrinsics = generate_camera_extrinsics(
            1, cfg.num_views
        ).squeeze(0)  # (V, 4, 4)

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed + idx)
        return generator

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h, w = self.cfg.image_size
        g = self._rng(idx)

        # Generate random images
        images = torch.randn(
            self.cfg.num_views,
            3,
            h,
            w,
            generator=g,
        )

        # Generate bounding boxes
        if self.cfg.use_3d_bbox:
            # 3D bounding boxes: (x, y, z, w, l, h, sin_yaw, cos_yaw)
            # x, y, z: 3D center position (normalized)
            # w, l, h: dimensions (normalized)
            # sin_yaw, cos_yaw: orientation
            boxes = torch.rand(
                self.cfg.num_queries,
                8,
                generator=g,
            )
            # Normalize to reasonable ranges
            boxes[:, :3] = boxes[:, :3] * 2 - 1  # x, y, z in [-1, 1]
            boxes[:, 3:6] = boxes[:, 3:6] * 0.5 + 0.1  # w, l, h in [0.1, 0.6]
            # sin_yaw, cos_yaw should satisfy sin^2 + cos^2 = 1
            yaw = torch.rand(self.cfg.num_queries, generator=g) * 2 * torch.pi
            boxes[:, 6] = torch.sin(yaw)
            boxes[:, 7] = torch.cos(yaw)
        else:
            # 2D bounding boxes: (cx, cy, w, h) normalized to [0, 1]
            boxes = torch.rand(
                self.cfg.num_queries,
                4,
                generator=g,
            )

        # Generate labels
        labels = torch.randint(
            0,
            self.cfg.num_classes,
            (self.cfg.num_queries,),
            generator=g,
        )

        result = {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "cam_intrinsics": self.cam_intrinsics.clone(),
            "cam_extrinsics": self.cam_extrinsics.clone(),
        }

        return result


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = torch.stack([item["images"] for item in batch])
    boxes = torch.stack([item["boxes"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    cam_intrinsics = torch.stack([item["cam_intrinsics"] for item in batch])
    cam_extrinsics = torch.stack([item["cam_extrinsics"] for item in batch])

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "cam_intrinsics": cam_intrinsics,
        "cam_extrinsics": cam_extrinsics,
    }


def build_dataloaders(cfg: Dict, distributed: bool):
    """
    Build training and validation dataloaders.

    Args:
        cfg: Configuration dictionary
        distributed: Whether to use distributed sampling

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        train_sampler: Training sampler (or None if not distributed)
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    use_3d_bbox = model_cfg.get("use_petr", False)

    synth_cfg = SyntheticConfig(
        image_size=tuple(data_cfg["image_size"]),
        num_views=int(data_cfg["num_views"]),
        num_classes=int(model_cfg["num_classes"]),
        num_queries=int(model_cfg["num_queries"]),
        seed=int(data_cfg.get("seed", 42)),
        use_3d_bbox=use_3d_bbox,
    )

    train_dataset = SyntheticMultiViewDataset(int(data_cfg["train_samples"]), synth_cfg)
    val_dataset = SyntheticMultiViewDataset(int(data_cfg["val_samples"]), synth_cfg)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=val_sampler,
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, train_sampler