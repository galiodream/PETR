"""
PETR: Position Embedding Transformation for Multi-View 3D Object Detection

Reference: PETR: Position Embedding Transformation for Multi-View 3D Object Detection
https://arxiv.org/abs/2203.05625

Key innovations:
1. 3D Position Embedding (3D PE): Projects 3D coordinates to image features
2. Multi-view feature aggregation via transformer encoder
3. Object queries for 3D detection
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


class SimpleBackbone(nn.Module):
    """ResNet50 + FPN backbone for feature extraction."""

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # C2 (stride=4)
        self.layer2 = backbone.layer2  # C3 (stride=8)
        self.layer3 = backbone.layer3  # C4 (stride=16)
        self.layer4 = backbone.layer4  # C5 (stride=32)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Keep stride=8 output to match previous token count/compute profile.
        pyramid_feats = self.fpn({"c2": c2, "c3": c3, "c4": c4, "c5": c5})
        return pyramid_feats["c3"]


class SinCosPositionEmbedding(nn.Module):
    """2D Sinusoidal Position Embedding (from DETR)."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        y_embed = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, h, 1).expand(b, h, w)
        x_embed = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, w).expand(b, h, w)

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos.permute(0, 3, 1, 2).contiguous()


class CameraAwarePositionEmbedding3D(nn.Module):
    """
    Camera-aware 3D position embedding.

    This approximates PETR's idea of lifting each image location to 3D points
    across multiple depth bins and encoding these coordinates.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_depth_bins: int = 8,
        depth_start: float = 1.0,
        depth_end: float = 50.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_depth_bins = num_depth_bins
        self.depth_start = depth_start
        self.depth_end = depth_end

        self.encoder = nn.Sequential(
            nn.Linear(3 * num_depth_bins, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        feat: torch.Tensor,  # (B, V, C, H, W)
        cam_intrinsics: torch.Tensor,  # (B, V, 3, 3)
        cam_extrinsics: torch.Tensor,  # (B, V, 4, 4), world -> cam
        image_size: Tuple[int, int],  # (H_img, W_img)
    ) -> torch.Tensor:
        B, V, _, H, W = feat.shape
        image_h, image_w = image_size
        device = feat.device
        dtype = feat.dtype

        if cam_intrinsics.shape[:2] != (B, V) or cam_extrinsics.shape[:2] != (B, V):
            raise ValueError("Camera tensor batch/view dims must match image batch/view dims.")

        grid_y, grid_x = torch.meshgrid(
            (torch.arange(H, device=device, dtype=dtype) + 0.5) * (float(image_h) / float(H)),
            (torch.arange(W, device=device, dtype=dtype) + 0.5) * (float(image_w) / float(W)),
            indexing="ij",
        )
        ones = torch.ones_like(grid_x)
        pixel_homo = torch.stack((grid_x, grid_y, ones), dim=-1).view(1, 1, H * W, 3)
        pixel_homo = pixel_homo.expand(B, V, -1, -1)  # (B, V, H*W, 3)

        k_inv = torch.inverse(cam_intrinsics)  # (B, V, 3, 3)
        rays_cam = torch.einsum("bvij,bvpj->bvpi", k_inv, pixel_homo)  # (B, V, H*W, 3)

        depth_bins = torch.linspace(
            self.depth_start,
            self.depth_end,
            self.num_depth_bins,
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, self.num_depth_bins, 1)
        points_cam = rays_cam.unsqueeze(-2) * depth_bins  # (B, V, H*W, D, 3)

        rotation = cam_extrinsics[:, :, :3, :3]
        translation = cam_extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(2)

        # x_cam = R @ x_world + t  ->  x_world(row) = (x_cam(row) - t(row)) @ R
        points_world = torch.einsum("bvpdj,bvjk->bvpdk", points_cam - translation, rotation)
        points_world = (points_world / max(self.depth_end, 1e-6)).clamp(min=-2.0, max=2.0)

        points_world = points_world.flatten(-2)  # (B, V, H*W, D*3)
        pos_3d = self.encoder(points_world)  # (B, V, H*W, d_model)
        return pos_3d.view(B, V * H * W, self.d_model)


class SimplePositionEmbedding3D(nn.Module):
    """
    Simplified 3D Position Embedding for training stability.

    Instead of full camera projection, we use learnable embeddings
    combined with spatial grid. This is more stable for synthetic data.
    """

    def __init__(self, d_model: int = 256, num_views: int = 6) -> None:
        super().__init__()
        self.d_model = d_model

        # Learnable view embeddings
        self.view_embed = nn.Embedding(num_views, d_model)

        # 2D position embedding
        self.pos_2d = SinCosPositionEmbedding(num_pos_feats=d_model // 2)

        # MLP to combine 2D and view embeddings into 3D-aware embedding
        self.combine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        feat: torch.Tensor,  # (B, V, C, H, W)
    ) -> torch.Tensor:
        """
        Generate 3D-aware position embeddings.

        Returns:
            pos: (B, V*H*W, C) position embeddings
        """
        B, V, C, H, W = feat.shape
        device = feat.device

        # 2D position embedding for each view
        feat_flat = feat.view(B * V, C, H, W)
        pos_2d = self.pos_2d(feat_flat)  # (B*V, C, H, W)
        pos_2d = pos_2d.view(B, V, C, H, W)

        # View embeddings
        view_ids = torch.arange(V, device=device)
        view_embed = self.view_embed(view_ids)  # (V, C)
        view_embed = view_embed.view(1, V, C, 1, 1).expand(B, -1, -1, H, W)

        # Combine
        pos_combined = pos_2d + view_embed  # (B, V, C, H, W)
        pos_combined = pos_combined.permute(0, 1, 3, 4, 2)  # (B, V, H, W, C)
        pos_combined = pos_combined.reshape(B, V * H * W, C)

        return self.combine(pos_combined)


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for DETR-like object detection.

    This class computes an assignment between the ground truth objects and the
    predictions of the network. For efficiency reasons, the targets don't include
    the no_object. Because of this, in general, there are more predictions than
    ground truth objects. Therefore, the predictions are matched with no_object
    first, then the matched predictions are paired with ground truth objects.

    Reference: End-to-End Object Detection with Transformers (DETR)
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, N, num_classes)
        pred_boxes: torch.Tensor,    # (B, N, 4) or (B, N, 7) for 3D
        tgt_labels: torch.Tensor,    # (B, M) where M is number of targets
        tgt_boxes: torch.Tensor,     # (B, M, 4) or (B, M, 7)
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Args:
            pred_logits: Class predictions (B, N, num_classes)
            pred_boxes: Box predictions (B, N, D) where D=4 for 2D, D=7 for 3D
            tgt_labels: Ground truth labels (B, M)
            tgt_boxes: Ground truth boxes (B, M, D)

        Returns:
            List of (pred_indices, tgt_indices) for each batch
        """
        B, N, num_classes = pred_logits.shape

        # Flatten to process all batches
        # pred_logits: (B, N, num_classes) -> (B*N, num_classes)
        # pred_boxes: (B, N, D) -> (B*N, D)

        cost_matrices = []
        for b in range(B):
            pred_logits_b = pred_logits[b]  # (N, num_classes)
            pred_boxes_b = pred_boxes[b]     # (N, D)
            tgt_labels_b = tgt_labels[b]     # (M,)
            tgt_boxes_b = tgt_boxes[b]        # (M, D)

            num_tgt = tgt_labels_b.shape[0]

            if num_tgt == 0:
                # No targets, all predictions are unmatched
                cost_matrices.append((torch.tensor([], device=pred_logits.device, dtype=torch.long),
                                      torch.tensor([], device=pred_logits.device, dtype=torch.long)))
                continue

            # Classification cost: use probability instead of log probability
            pred_probs = pred_logits_b.softmax(-1)  # (N, num_classes)
            cost_class = -pred_probs[:, tgt_labels_b]  # (N, M)

            # BBox L1 cost
            cost_bbox = torch.cdist(pred_boxes_b, tgt_boxes_b, p=1)  # (N, M)

            # Giou cost (for 2D boxes, first 4 elements)
            if pred_boxes_b.shape[-1] >= 4 and tgt_boxes_b.shape[-1] >= 4:
                pred_boxes_2d = pred_boxes_b[:, :4]  # (N, 4)
                tgt_boxes_2d = tgt_boxes_b[:, :4]    # (M, 4)
                cost_giou = -self._compute_giou(pred_boxes_2d, tgt_boxes_2d)  # (N, M)
            else:
                cost_giou = 0

            # Final cost matrix
            cost = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )  # (N, M)

            # Hungarian matching
            pred_indices, tgt_indices = self._hungarian_matching(cost)
            cost_matrices.append((pred_indices, tgt_indices))

        return cost_matrices

    def _compute_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute GIoU between two sets of boxes.
        Boxes are in format (cx, cy, w, h) normalized to [0, 1].

        Args:
            boxes1: (N, 4)
            boxes2: (M, 4)

        Returns:
            giou: (N, M)
        """
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        def box_cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)

        boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)  # (N, 4)
        boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)  # (M, 4)

        # Compute intersection
        lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # (N, M, 2)
        rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # (N, M, 2)

        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

        # Compute areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])  # (N,)
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])  # (M,)

        # Compute union
        union = area1[:, None] + area2[None, :] - inter  # (N, M)

        # Compute IoU
        iou = inter / (union + 1e-6)  # (N, M)

        # Compute enclosing box
        lt_enc = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # (N, M, 2)
        rb_enc = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # (N, M, 2)

        wh_enc = rb_enc - lt_enc  # (N, M, 2)
        area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]  # (N, M)

        # Compute GIoU
        giou = iou - (area_enc - union) / (area_enc + 1e-6)  # (N, M)

        return giou

    def _hungarian_matching(self, cost_matrix: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform Hungarian matching on cost matrix.

        Args:
            cost_matrix: (N, M) cost matrix

        Returns:
            pred_indices: matched prediction indices
            tgt_indices: matched target indices
        """
        try:
            from scipy.optimize import linear_sum_assignment
            cost_np = cost_matrix.cpu().numpy()
            pred_indices, tgt_indices = linear_sum_assignment(cost_np)
            return (
                torch.from_numpy(pred_indices).to(cost_matrix.device),
                torch.from_numpy(tgt_indices).to(cost_matrix.device),
            )
        except ImportError:
            # Fallback to greedy matching if scipy is not available
            pred_indices = []
            tgt_indices = []
            cost = cost_matrix.clone()
            num_pred, num_tgt = cost.shape

            for _ in range(min(num_pred, num_tgt)):
                idx = cost.argmin()
                pred_idx = idx // num_tgt
                tgt_idx = idx % num_tgt
                pred_indices.append(pred_idx.item())
                tgt_indices.append(tgt_idx.item())
                cost[pred_idx, :] = float('inf')
                cost[:, tgt_idx] = float('inf')

            return (
                torch.tensor(pred_indices, device=cost_matrix.device),
                torch.tensor(tgt_indices, device=cost_matrix.device),
            )


class SETCriterion(nn.Module):
    """
    Loss criterion with Hungarian matching for PETR.

    This is the loss function used in DETR and PETR.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_ce: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.eos_coef = eos_coef

        # Class weights for handling imbalance (no-object class gets lower weight)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, N, num_classes)
        pred_boxes: torch.Tensor,    # (B, N, D)
        tgt_labels: torch.Tensor,    # (B, M) with values in [0, num_classes)
        tgt_boxes: torch.Tensor,     # (B, M, D)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute losses with Hungarian matching.

        Returns:
            loss: total loss
            loss_dict: dictionary of individual losses
        """
        B, N, _ = pred_logits.shape

        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, tgt_labels, tgt_boxes)

        # Initialize losses
        loss_ce = torch.tensor(0.0, device=pred_logits.device)
        loss_bbox = torch.tensor(0.0, device=pred_logits.device)
        loss_giou = torch.tensor(0.0, device=pred_logits.device)

        num_matches = 0

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            num_matches += len(pred_idx)

            # Classification loss for matched predictions
            pred_logits_matched = pred_logits[b, pred_idx]  # (num_matched, num_classes)
            tgt_labels_matched = tgt_labels[b, tgt_idx]      # (num_matched,)

            # Cross entropy loss
            loss_ce += F.cross_entropy(pred_logits_matched, tgt_labels_matched, reduction='sum')

            # BBox L1 loss
            pred_boxes_matched = pred_boxes[b, pred_idx]  # (num_matched, D)
            tgt_boxes_matched = tgt_boxes[b, tgt_idx]      # (num_matched, D)
            loss_bbox += F.l1_loss(pred_boxes_matched, tgt_boxes_matched, reduction='sum')

            # GIoU loss (for 2D boxes, first 4 elements)
            if pred_boxes_matched.shape[-1] >= 4:
                giou = self.matcher._compute_giou(pred_boxes_matched[:, :4], tgt_boxes_matched[:, :4])
                loss_giou += (1 - giou.diag()).sum()

        # Normalize by number of matches
        num_matches = max(num_matches, 1)
        loss_ce = loss_ce / num_matches
        loss_bbox = loss_bbox / num_matches
        loss_giou = loss_giou / num_matches

        # Total loss
        loss = self.weight_ce * loss_ce + self.weight_bbox * loss_bbox + self.weight_giou * loss_giou

        return loss, {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }


class PETRHead(nn.Module):
    """
    PETR Detection Head.

    Predicts class labels and 3D bounding boxes from decoder output.
    """

    def __init__(self, d_model: int, num_classes: int, num_queries: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Classification head
        self.class_embed = nn.Linear(d_model, num_classes)

        # 3D bounding box head: (x, y, z, w, l, h, sin_yaw, cos_yaw)
        # We predict 8 values: x, y, z, w, l, h, sin(yaw), cos(yaw)
        self.bbox_embed = MLP(d_model, d_model, 8, num_layers=3)

    def forward(self, hs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hs: Decoder output (B, N, d_model)

        Returns:
            pred_logits: Class logits (B, N, num_classes)
            pred_boxes: 3D boxes (B, N, 8) - (x, y, z, w, l, h, sin_yaw, cos_yaw)
        """
        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs)

        # Normalize boxes
        # x, y, z: use sigmoid to constrain to [0, 1] range (normalized coordinates)
        # w, l, h: use sigmoid for normalized sizes
        # sin_yaw, cos_yaw: already in valid range after MLP
        pred_boxes[..., :6] = pred_boxes[..., :6].sigmoid()

        return pred_logits, pred_boxes


class PETR(nn.Module):
    """
    PETR: Position Embedding Transformation for Multi-View 3D Object Detection.

    This implementation follows the paper:
    "PETR: Position Embedding Transformation for Multi-View 3D Object Detection"
    https://arxiv.org/abs/2203.05625

    Architecture:
    1. Multi-view image backbone (shared weights)
    2. 3D Position Embedding generation
    3. Feature + 3D PE encoding via transformer encoder
    4. Object query decoding via transformer decoder
    5. Detection head for classification and 3D bbox regression
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_queries: int,
        num_classes: int,
        num_views: int,
        backbone_channels: int,
        with_3d_pe: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_views = num_views
        self.with_3d_pe = with_3d_pe

        # Backbone
        self.backbone = SimpleBackbone(out_channels=backbone_channels)
        self.input_proj = nn.Conv2d(backbone_channels, d_model, kernel_size=1)

        # Position embeddings
        self.pos_3d = CameraAwarePositionEmbedding3D(d_model=d_model)
        self.pos_3d_simple = SimplePositionEmbedding3D(d_model=d_model, num_views=num_views)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Query embeddings (learnable)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Detection head
        self.head = PETRHead(d_model, num_classes, num_queries)

        # Loss criterion
        self.matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=0.0)
        self.criterion = SETCriterion(num_classes, self.matcher, weight_giou=0.0)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        images: torch.Tensor,
        cam_intrinsics: Optional[torch.Tensor] = None,
        cam_extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Multi-view images (B, V, 3, H, W)
            cam_intrinsics: Camera intrinsics (B, V, 3, 3), required when with_3d_pe=True
            cam_extrinsics: Camera extrinsics (B, V, 4, 4), required when with_3d_pe=True

        Returns:
            dict with keys:
                pred_logits: (B, N, num_classes) class predictions
                pred_boxes: (B, N, 8) 3D box predictions
        """
        B, V, _, H, W = images.shape

        # Extract features from each view
        x = images.flatten(0, 1)  # (B*V, 3, H, W)
        feats = self.backbone(x)  # (B*V, C', H', W')
        feats = self.input_proj(feats)  # (B*V, d_model, H', W')

        _, C, h, w = feats.shape

        # Reshape for multi-view processing
        feats = feats.view(B, V, C, h, w)

        # Get 3D position embeddings
        if self.with_3d_pe:
            if cam_intrinsics is None or cam_extrinsics is None:
                raise ValueError("cam_intrinsics and cam_extrinsics are required when with_3d_pe=True.")
            pos_3d = self.pos_3d(
                feats,
                cam_intrinsics=cam_intrinsics,
                cam_extrinsics=cam_extrinsics,
                image_size=(H, W),
            )
        else:
            pos_3d = self.pos_3d_simple(feats)

        # Flatten spatial dimensions
        src = feats.permute(0, 1, 3, 4, 2).reshape(B, V * h * w, C)  # (B, V*h*w, d_model)

        # Encode features with 3D position embeddings
        memory = self.encoder(src + pos_3d)  # (B, V*h*w, d_model)

        # Decode object queries
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)
        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt=tgt + query, memory=memory)  # (B, N, d_model)

        # Predict
        pred_logits, pred_boxes = self.head(hs)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss with Hungarian matching.

        Args:
            outputs: Model outputs with pred_logits and pred_boxes
            targets: Ground truth with labels and boxes

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        return self.criterion(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            targets["labels"],
            targets["boxes"],
        )


class PETRLite(nn.Module):
    """
    Lightweight PETR implementation for quick testing and experimentation.

    This is a simplified version with:
    - Simple 2D position embedding instead of full 3D PE
    - 2D bbox prediction instead of 3D
    - Direct loss computation without Hungarian matching

    For full PETR implementation, use the PETR class above.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_queries: int,
        num_classes: int,
        num_views: int,
        backbone_channels: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.backbone = SimpleBackbone(out_channels=backbone_channels)
        self.input_proj = nn.Conv2d(backbone_channels, d_model, kernel_size=1)
        self.position_embedding = SinCosPositionEmbedding(num_pos_feats=d_model // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.view_embed = nn.Embedding(num_views, d_model)

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        images: torch.Tensor,
        cam_intrinsics: Optional[torch.Tensor] = None,
        cam_extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, v, _, _, _ = images.shape
        x = images.flatten(0, 1)
        feats = self.backbone(x)
        feats = self.input_proj(feats)

        pos = self.position_embedding(feats)

        _, c, h, w = feats.shape
        feats = feats.view(b, v, c, h, w)
        pos = pos.view(b, v, c, h, w)

        view_ids = torch.arange(v, device=images.device)
        view_tokens = self.view_embed(view_ids).view(1, v, c, 1, 1)
        src = feats + pos + view_tokens

        src = src.permute(0, 1, 3, 4, 2).reshape(b, v * h * w, c)
        memory = self.encoder(src)

        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt=tgt + query, memory=memory)

        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }


def build_model(cfg: Dict) -> nn.Module:
    """Build PETR model from config."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # Check if we should use full PETR with 3D PE
    use_petr = model_cfg.get("use_petr", False)

    if use_petr:
        return PETR(
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_encoder_layers=int(model_cfg["num_encoder_layers"]),
            num_decoder_layers=int(model_cfg["num_decoder_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            num_queries=int(model_cfg["num_queries"]),
            num_classes=int(model_cfg["num_classes"]),
            num_views=int(data_cfg["num_views"]),
            backbone_channels=int(model_cfg["backbone_channels"]),
            with_3d_pe=model_cfg.get("with_3d_pe", True),
        )
    else:
        return PETRLite(
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_encoder_layers=int(model_cfg["num_encoder_layers"]),
            num_decoder_layers=int(model_cfg["num_decoder_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            num_queries=int(model_cfg["num_queries"]),
            num_classes=int(model_cfg["num_classes"]),
            num_views=int(data_cfg["num_views"]),
            backbone_channels=int(model_cfg["backbone_channels"]),
        )
