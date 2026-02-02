# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree
"""Model to extract BLIP embeddings from an image and its segmentation."""

from semantic_inference_python.config import Config, config_field

import torch
from torch import nn
import torchvision
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette, Color
import rospy
import time
import rospy

from typing import Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box with center and orientation."""

    world_P_center: torch.Tensor
    world_R_center: torch.Tensor
    dimensions: torch.Tensor

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the bounding box.
        :param points: [N, 3] tensor of points in world coordinates
        :return: [N] boolean tensor
        """
        # Translate so that box center is at origin
        rel_points = points - self.world_P_center

        # Convert points from world to local coordinates
        # (transpose rotation matrix to invert world ← local)
        local_points = rel_points @ self.world_R_center.T

        # Check if within half dimensions along each axis
        half_dims = self.dimensions / 2.0
        inside_mask = (local_points.abs() <= half_dims).all(dim=1)

        return inside_mask

    def centroid(self) -> torch.Tensor:
        """Get the centroid of the bounding box."""
        return self.world_P_center


@dataclass(frozen=True)
class ActiveRelations:
    """Active relations between objects."""

    bounding_boxes: List[BoundingBox]
    relationships: Set[Tuple[int, int]]


@dataclass
class VLMFeatureExtractorConfig(Config):
    """Main configuration for VLM feature extractor."""

    vlm: Any = config_field("vlm", default="instruct_blip_visual")
    max_batch_vlm: int = 3
    min_area: int = 10
    min_depth: float = 0.05
    dense_representation_radius_m: float = 10
    full_img: bool = False
    crop_img_to_bb: bool = False
    publish_masks: bool = False
    object_labels: List[int] = field(default_factory=list)
    cuda: bool = True
    verbose: bool = False
    include_bbs: bool = False
    use_graph_edges: bool = True


class VLMFeatureExtractor(nn.Module):
    """Module to extract VLM embeddings from an image and its segmentation."""

    def __init__(
        self, config: VLMFeatureExtractorConfig, colors_to_labels=None, id_to_name=None
    ):
        """Construct a VLM feature extractor."""
        super(VLMFeatureExtractor, self).__init__()
        # for detecting model device
        self._device = torch.device("cpu")

        self.config = config
        self.colors_to_labels = colors_to_labels
        self.id_to_name = id_to_name
        if not self.config.full_img or self.config.crop_img_to_bb:
            self.config.vlm.cropping = False
        self.encoder = self.config.vlm.create()

        # Annotation tools
        self.box_annotator = sv.BoxAnnotator(
            text_scale=0.3,
            text_thickness=1,
            text_padding=2,
            color=ColorPalette.DEFAULT,
        )
        self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)

        self._active_relations = None

    @classmethod
    def construct(cls, **kwargs):
        """Construct a VLM feature extractor."""
        config = VLMFeatureExtractorConfig(**kwargs)
        config.update(kwargs)
        return cls(config)

    @property
    def device(self):
        """Get the device of the model."""
        return self._device

    @device.setter
    def device(self, device):
        """Set the device of the model."""
        self._device = device

    @property
    def active_relations(self):
        """Get the active relations."""
        return self._active_relations

    @active_relations.setter
    def active_relations(self, relations: Optional[ActiveRelations] = None):
        """Set the active relations."""
        self._active_relations = relations

    def move_to(self, device):
        """Move the model to a device."""
        self.encoder.move_to(device)
        self.device = device

    def get_masks_from_segmentation(
        self, seg_img: np.ndarray, panoptic_img: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get masks from panoptic segmentation and their corresponding
        colors from the segmentation image.

        :param seg_img: Segmentation image
        :param panoptic_img: Panoptic image
        :return: Tuple of masks and colors
        """
        # Get unique panptic ids
        unique_ids, indices = np.unique(panoptic_img, return_index=True)

        # Get masks using unique ids
        masks = torch.from_numpy(panoptic_img)[:, :, None] == torch.from_numpy(
            unique_ids
        )

        return (
            masks.permute(2, 0, 1).to(torch.bool).to(self.device),
            np.reshape(seg_img, (-1, 3))[indices],
        )

    def _graph_aware_selection(
        self,
        bboxes: torch.Tensor,
        masks: torch.Tensor,
        depth: torch.Tensor,
        camera_matrix: torch.Tensor,
        transform: torch.Tensor,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Optimized: Select bounding boxes and combinations prioritizing pairs NOT connected in the active graph.
        Leverages spatial locality and vectorization for speed.

        Returns:
            choice (np.ndarray): indices of chosen detections
            combs (torch.Tensor): [M, 2] index pairs into choice
        """
        # Quick sanity check
        if (
            self.active_relations is None
            or not self.config.use_graph_edges
            or len(self.active_relations.bounding_boxes) == 0
            or depth is None
            or camera_matrix is None
            or transform is None
        ):
            choice = np.random.choice(
                bboxes.shape[0], self.config.max_batch_vlm, replace=False
            )
            combs = torch.combinations(torch.arange(len(choice)))
            return choice, combs

        device = self.device
        N = bboxes.shape[0]

        # ---- 1) Compute 3D centroids in world coordinates ----
        K = camera_matrix.to(device).float()
        Kinv = torch.inverse(K)
        if depth.dim() == 3:
            depth = depth.squeeze()

        det_world_centroids = []
        for m in masks:  # m: [H, W]
            ys, xs = torch.nonzero(m, as_tuple=True)
            if ys.numel() == 0:
                det_world_centroids.append(None)
                continue

            u = xs.float().mean()
            v = ys.float().mean()
            z = depth[ys, xs].float().mean()

            if not torch.isfinite(z) or z <= 0:
                det_world_centroids.append(None)
                continue

            pix = torch.tensor([u, v, 1.0], device=device)
            cam_P = (Kinv @ pix) * z
            cam_P_h = torch.cat([cam_P, torch.ones(1, device=device)], dim=0)
            world_P_h = transform @ cam_P_h
            world_P = world_P_h[:3] / world_P_h[3].clamp(min=1e-8)
            det_world_centroids.append(world_P)

        # ---- 2) Assign to nearby graph nodes ----
        radius = self.config.dense_representation_radius_m + 2.0
        graph_centroids = torch.stack(
            [bb.centroid() for bb in self.active_relations.bounding_boxes]
        ).to(
            device
        )  # [M,3]
        det_to_node = []
        dists = torch.norm(graph_centroids - transform[:3, 3], dim=1)
        nearby_idx = torch.nonzero(dists <= radius, as_tuple=False).view(-1)
        for P in det_world_centroids:
            if P is None:
                det_to_node.append(-1)
                continue
            found = -1
            for j in nearby_idx:
                if self.active_relations.bounding_boxes[int(j)].contains(
                    P.view(1, 3).double()
                )[0]:
                    found = int(j)
                    break
            det_to_node.append(found)
        det_to_node = torch.tensor(det_to_node, device=device, dtype=torch.int64)

        # ---- 3) Rank detections by degree in the graph ----
        present_nodes = set([int(n.item()) for n in det_to_node if n >= 0])
        neighbors = {n: 0 for n in present_nodes}
        for a, b in self.active_relations.relationships:
            if a in present_nodes and b in present_nodes:
                neighbors[a] += 1
                neighbors[b] += 1

        degrees = torch.tensor(
            [
                0
                if det_to_node[i] < 0
                else max(neighbors.get(int(det_to_node[i].item()), 0), 0)
                for i in range(N)
            ],
            device=device,
        )
        # If all elements are the same, return random choice
        min_degrees = degrees.min()
        # Sample self.max_batch_vlm - 1 elements with the minimum degree
        min_degree_indices = torch.nonzero(degrees == min_degrees, as_tuple=False).view(
            -1
        )
        if min_degree_indices.numel() > self.config.max_batch_vlm - 1:
            sub_choice = np.random.choice(
                min_degree_indices.cpu().numpy(),
                self.config.max_batch_vlm - 1,
                replace=False,
            )
        else:
            sub_choice = min_degree_indices.cpu().numpy()
        # Sample one more element randomly (from the remaining elements --> torch.arange(N) - sub_choice)
        remaining_indices = torch.tensor(
            list(set(torch.arange(N).cpu().numpy()) - set(sub_choice)), device=device
        )
        random_choice = np.random.choice(
            remaining_indices.cpu().numpy(),
            1,
            replace=False,
        )
        choice = np.concatenate((sub_choice, random_choice))

        # sorted_idx = torch.argsort(degrees, dim=0, stable=True)
        # keep = min(self.config.max_batch_vlm, N)
        # choice = sorted_idx[:keep].cpu().numpy()

        # ---- 4) Build combinations with graph edges at the end ----
        combs = torch.combinations(torch.arange(len(choice)))
        return choice, combs
        # if combs.numel() > 0:
        #     local_nodes = det_to_node[torch.from_numpy(choice).to(device)]
        #     i_idx, j_idx = combs[:, 0], combs[:, 1]
        #     ni, nj = local_nodes[i_idx], local_nodes[j_idx]

        #     valid_mask = (ni >= 0) & (nj >= 0)
        #     valid_idx = torch.nonzero(valid_mask).view(-1)
        #     edge_set = set((min(a, b), max(a, b)) for a, b in self.active_relations.relationships)
        #     edge_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

        #     for idx in valid_idx:
        #         a, b = int(ni[idx]), int(nj[idx])
        #         if (min(a, b), max(a, b)) in edge_set:
        #             edge_mask[idx] = True

        #             return choice, combs

    def encode(
        self,
        img: torch.Tensor,
        depth: torch.Tensor,
        masks: torch.Tensor,
        colors: np.ndarray,
        choice: Optional[np.ndarray] = None,
        camera_matrix: Optional[torch.Tensor] = None,
        transform: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Encode image with VLM embeddings.

        Args:
            img (torch.Tensor): Image tensor
            depth (torch.Tensor): Depth tensor
            masks (torch.Tensor): Masks
            colors (np.ndarray): Colors of the masks
            choice (np.ndarray): Chosen bounding boxes
            camera_matrix (Optional[torch.Tensor]): Camera matrix
            transform (Optional[torch.Tensor]): Transform from camera to world frame
        Returns:

            torch.Tensor: Encoded image, chosen bounding boxes, optional annotated image
        """
        bboxes = torchvision.ops.masks_to_boxes(masks).to(torch.int32)
        if self.config.full_img:
            choice = np.arange(masks.shape[0])
            self.box_annotator.color = ColorPalette(
                [Color(int(c[0]), int(c[1]), int(c[2])) for c in colors]
            )
            self.mask_annotator.color = self.box_annotator.color
            if self.config.crop_img_to_bb:
                union_imgs = (
                    [
                        img[
                            bboxes[choice, 1]
                            .min()
                            .item() : bboxes[choice, 3]
                            .max()
                            .item(),
                            bboxes[choice, 0]
                            .min()
                            .item() : bboxes[choice, 2]
                            .max()
                            .item(),
                            ...,
                        ]
                    ]
                    if masks.shape[0] > 1
                    else []
                )
            else:
                union_imgs = [img] if masks.shape[0] > 1 else []
            combs = torch.combinations(torch.arange(len(choice)))
        else:
            if not choice and bboxes.shape[0] > self.config.max_batch_vlm:
                if not self.config.use_graph_edges:
                    choice = np.random.choice(
                        bboxes.shape[0], self.config.max_batch_vlm, replace=False
                    )
                    combs = torch.combinations(torch.arange(len(choice)))

                else:
                    start_time = time.time()
                    choice, combs = self._graph_aware_selection(
                        bboxes, masks, depth, camera_matrix, transform
                    )
                    if self.config.verbose:
                        rospy.loginfo(
                            f"[VLM node] Graph-aware selection time: {(time.time() - start_time) * 1000:.3f} ms"
                        )
            elif not choice:
                choice = np.arange(bboxes.shape[0])
                combs = torch.combinations(torch.arange(len(choice)))
            chosen_bboxes = bboxes[torch.from_numpy(choice)]
            chosen_colors = colors[choice]
            self.box_annotator.color = ColorPalette(
                [Color(int(c[2]), int(c[1]), int(c[0])) for c in chosen_colors]
            )
            self.mask_annotator.color = self.box_annotator.color
            if self.config.include_bbs:
                annotated_imgs = [
                    self.box_annotator.annotate(
                        scene=img.cpu().clone().numpy(),
                        detections=replace(
                            sv.Detections(xyxy=bboxes.cpu().numpy()[choice[comb]])
                        ),
                        skip_label=True,
                    )
                    for comb in combs.cpu()
                ]
                pad = 20
                union_imgs = [
                    annotated_imgs[k][
                        max(chosen_bboxes[[j, i], 1].min().item() - pad, 0) : min(
                            chosen_bboxes[[j, i], 3].max().item() + pad,
                            annotated_imgs[k].shape[0],
                        ),
                        max(chosen_bboxes[[j, i], 0].min().item() - pad, 0) : min(
                            chosen_bboxes[[j, i], 2].max().item() + pad,
                            annotated_imgs[k].shape[1],
                        ),
                        ...,
                    ]
                    for k, (i, j) in enumerate(combs)
                ]
            else:
                union_imgs = [
                    img[
                        chosen_bboxes[[j, i], 1]
                        .min()
                        .item() : chosen_bboxes[[j, i], 3]
                        .max()
                        .item(),
                        chosen_bboxes[[j, i], 0]
                        .min()
                        .item() : chosen_bboxes[[j, i], 2]
                        .max()
                        .item(),
                        ...,
                    ]
                    for i, j in combs
                ]
        if len(union_imgs) == 0:
            if self.config.verbose:
                rospy.logwarn("[VLM] No images found")
            return None, None, None

        annotated_image = None
        if self.config.publish_masks:
            if self.config.full_img:
                chosen_bboxes = bboxes
            curr_det = sv.Detections(
                xyxy=chosen_bboxes.cpu().numpy(),
                mask=masks[torch.from_numpy(choice)].cpu().numpy(),
            )
            curr_det = replace(curr_det)
            curr_det.class_id = np.arange(len(curr_det))
            annotated_image = self.mask_annotator.annotate(
                scene=img.cpu().numpy().copy(), detections=curr_det
            )
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=curr_det
            )
        start_time = time.time()
        with torch.inference_mode():
            res = self.encoder.encode_images(union_imgs).cpu()
        if self.config.verbose:
            rospy.loginfo(
                f"[VLM node] BLIP inference time: {(time.time() - start_time) * 1000:.3f} ms"
            )
        return (
            res,
            choice[combs.reshape(-1).cpu().numpy().astype(int)],
            annotated_image,
        )

    def forward(
        self,
        img: np.ndarray,
        seg_img: np.ndarray,
        panoptic_img: np.ndarray,
        depth_img: np.ndarray = None,
        choice: np.ndarray = None,
        camera_matrix: Optional[np.ndarray] = None,
        transform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract masks, boxes and labels and encode them.

        Args:
            img (np.array): Image
            seg_img (np.array): Segmentation image
            panoptic_img (np.array): Panoptic image
            depth_img (np.array): Depth image
            choice (np.ndarray): Chosen bounding boxes
            camera_matrix (Optional[np.ndarray]): Camera matrix for undistortion

        Returns:
            Results: Encoded image, panoptic ids
        """
        masks, colors = self.get_masks_from_segmentation(seg_img, panoptic_img)
        # Log time in ms
        areas = torchvision.ops.box_area(torchvision.ops.masks_to_boxes(masks))

        if not np.any(depth_img):
            depth_img = None
        mean_depths = [
            depth_img[mask.cpu().numpy().astype(bool)].mean()
            if depth_img is not None
            else self.config.min_depth + 0.1
            for mask in masks
        ]
        masks_colors = [
            (mask.unsqueeze(0), color)
            for mask, color, area, mean_depth in zip(masks, colors, areas, mean_depths)
            if int(
                self.colors_to_labels[
                    tuple(seg_img[mask.cpu().numpy().astype(bool)][0])
                ]
            )
            in self.config.object_labels
            and area > self.config.min_area
            and (
                mean_depth > self.config.min_depth
                and mean_depth < self.config.dense_representation_radius_m
            )
        ]
        if len(masks_colors) == 0:
            if self.config.verbose:
                rospy.logwarn("[VLM] No masks found")
            return None, None, None
        masks = torch.cat([mask for mask, _ in masks_colors], dim=0)
        colors = np.array([color for _, color in masks_colors])
        features, choice, annot_img = self.encode(
            torch.from_numpy(img).to(self.device),
            torch.from_numpy(depth_img).to(self.device),
            masks,
            colors,
            choice,
            torch.from_numpy(camera_matrix).to(self.device)
            if camera_matrix is not None
            else None,
            transform,
        )

        return (
            features,
            np.array(
                [
                    panoptic_img[mask.astype(bool)][0]
                    for mask in masks.cpu().numpy()[choice]
                ]
                if features is not None
                else None
            ),
            annot_img,
        )
