#!/usr/bin/env python3
"""Recoloring module."""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from spark_config import Config
from supervision.draw.color import Color, ColorPalette

from semantic_inference_python import Logger


def parse_labels_file(labels_file_path: Path) -> tuple[int, dict[str, int], int]:
    """Parse a labels YAML file to get name to ID mapping and unknown ID.

    :param labels_file_path: Path to the labels YAML file
    :return: Tuple of total semantic labels, name to ID mapping, and unknown ID
    """

    with labels_file_path.open("r") as file:
        labels_data = yaml.safe_load(file)

    name_to_id = dict()
    unknown_id = -1

    for label in labels_data.get("label_names", []):
        name = label.get("name")
        obj_id = label.get("label")
        if name == "unknown":
            unknown_id = obj_id
            continue
        if name == "free":
            continue
        name_to_id[name] = obj_id
    name_to_id["invalid"] = -1

    return labels_data.get("total_semantic_labels", 0), name_to_id, unknown_id


def parse_csv_to_mappings(
    csv_file_path: Path,
) -> tuple[dict[int, tuple[int, int, int, int]], dict[str, int]]:
    """Parse a CSV file to mappings.

    :param csv_file_path: Path to the CSV file
    :return: Tuple of ID to RGBA and ID to color name mappings
    """
    # Dictionaries for mappings
    id_to_rgba = dict()
    id_to_color_name = dict()

    # Open and parse the CSV file
    with csv_file_path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Extract name, id, and RGBA values
            rgba = (
                int(row["red"]),
                int(row["green"]),
                int(row["blue"]),
                int(row["alpha"]),
            )
            obj_id = int(row["id"])
            color_name = "unknown"
            if "color" in row:
                color_name = str(row["color"])

            # Populate mappings
            if rgba == (255, 255, 255, 255):
                continue
            id_to_rgba[obj_id] = rgba
            id_to_color_name[obj_id] = color_name
    id_to_rgba[-1] = (0, 0, 0, 0)

    return id_to_rgba, id_to_color_name


@dataclass
class RecolorConfig(Config):
    """Configuration for recoloring."""

    colormap_path: Path = Path("")
    labels_path: Path = Path("")

    def initialize(self):
        """Initialize the configuration."""
        self.colormap_path = Path(self.colormap_path)
        self.labels_path = Path(self.labels_path)
        assert self.colormap_path.exists(), "Colormap file does not exist"
        assert self.labels_path.exists(), "Labels file does not exist"


class Recolor:
    """Recoloring class."""

    def __init__(self, config: RecolorConfig) -> None:
        """Initialize the recoloring class."""
        self.config = config
        self.config.initialize()

        # Parse the colormap file
        self.num_max_labels, self.name_to_id, self.unknown = parse_labels_file(
            self.config.labels_path
        )
        self.id_to_rgba, self.id_to_color_name = parse_csv_to_mappings(
            self.config.colormap_path
        )

    def add_labels(self, new_labels: list[str]) -> bool:
        if len(new_labels) == 0:
            return
        if len(new_labels) + len(self.name_to_id) > self.num_max_labels:
            Logger.info(
                f"Cannot add {len(new_labels)} new labels. "
                f"Max labels: {self.num_max_labels}, "
                f"Current labels: {len(self.name_to_id)}"
            )
            return False
        for label in new_labels:
            if label not in self.name_to_id:
                new_id = max(self.name_to_id.values()) + 1
                self.name_to_id[label] = new_id
        return True

    def get_color_from_name(self, name: str) -> tuple[int, int, int, int]:
        """
        Get the RGBA color for a given name.

        :param name: Name of the object
        :return: RGBA color tuple
        """
        if name in self.name_to_id:
            obj_id = self.name_to_id[name]
            return self.id_to_rgba[obj_id]
        else:
            return (0, 0, 0, 0)  # Default to black if name not found

    def get_color_name_from_name(self, name: str) -> str:
        """
        Get the color name for a given name.

        :param name: Name of the object
        :return: Color name string
        """
        if name in self.name_to_id:
            obj_id = self.name_to_id[name]
            return self.id_to_color_name[obj_id]
        else:
            return "unknown"

    def get_colorpalette(self, bgr=True) -> ColorPalette:
        """Get the color palette based on the recoloring configuration."""
        colors = [Color(rgba[2], rgba[1], rgba[0]) for rgba in self.id_to_rgba.values()]
        return ColorPalette(colors)

    def recolor_panoptic(self, panoptic: torch.Tensor) -> torch.Tensor:
        """
        Recolor a panoptic image based on the labels.

        :param panoptic: panoptic segmentation image
        :return: Recolored image (H, W, C)
        """
        recolored_img = torch.zeros(
            (panoptic.size(1), panoptic.size(2), 3), dtype=torch.uint8
        )
        unique_labels = torch.unique(panoptic)
        for label in unique_labels:
            r, g, b, _ = self.id_to_rgba[label.item()]
            recolored_img[panoptic == label] = torch.tensor(
                [r, g, b], dtype=torch.uint8
            )

        return recolored_img

    def recolor_image(self, masks: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Recolor an image based on masks and labels.

        :param masks: Masks for objects (N, H, W)
        :param labels: Labels for objects (N,)
        :return: Recolored image (H, W, C)
        """
        # Copy the image
        recolored_img = torch.zeros(
            (masks.size(1), masks.size(2), 3), dtype=torch.uint8
        )

        # Ensure masks are boolean and labels match the number of masks
        assert masks.shape[0] == labels.shape[0], (
            "Number of masks and labels must match"
        )
        assert masks.dtype == torch.bool, "Masks should be boolean arrays"

        # Iterate over each mask and corresponding label
        for mask, label in zip(masks, labels):
            if label.item() in self.id_to_rgba:
                # Retrieve the RGBA color for the label
                r, g, b, _ = self.id_to_rgba[label.item()]

                # Apply the color where the mask is True
                recolored_img[mask] = torch.tensor([r, g, b], dtype=torch.uint8)

        return recolored_img
