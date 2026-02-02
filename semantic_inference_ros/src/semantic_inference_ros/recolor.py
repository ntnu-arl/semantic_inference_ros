#!/usr/bin/env python3
"""Recoloring module."""
from pathlib import Path
import csv
from typing import Dict, Tuple

import torch

from semantic_inference_python import Config
from supervision.draw.color import Color, ColorPalette
from dataclasses import dataclass


def parse_csv_to_mappings(
    csv_file_path: Path,
) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[str, int]]:
    """Parse a CSV file to mappings.

    :param csv_file_path: Path to the CSV file
    :return: Tuple of ID to RGBA and NAME to ID mappings
    """
    # Dictionaries for mappings
    name_to_id = dict()
    id_to_rgba = dict()
    id_to_color_name = dict()

    # Open and parse the CSV file
    with csv_file_path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Extract name, id, and RGBA values
            name = row["name"]
            rgba = (
                int(row["red"]),
                int(row["green"]),
                int(row["blue"]),
                int(row["alpha"]),
            )
            obj_id = int(row["id"])
            color_name = str(row["color"])

            # Populate mappings
            if rgba == (255, 255, 255, 255):
                continue
            name_to_id[name] = obj_id
            id_to_rgba[obj_id] = rgba
            id_to_color_name[obj_id] = color_name
    id_to_rgba[-1] = (0, 0, 0, 0)
    name_to_id["invalid"] = -1

    return id_to_rgba, name_to_id, id_to_color_name


@dataclass
class RecolorConfig(Config):
    """Configuration for recoloring."""

    colormap_path: Path = Path("")

    def initialize(self):
        """Initialize the configuration."""
        self.colormap_path = Path(self.colormap_path)
        assert self.colormap_path.exists(), "Colormap file does not exist"


class Recolor:
    """Recoloring class."""

    def __init__(self, config: RecolorConfig) -> None:
        """Initialize the recoloring class."""
        self.config = config
        self.config.initialize()

        # Parse the colormap file
        self.id_to_rgba, self.name_to_id, self.id_to_color_name = parse_csv_to_mappings(
            self.config.colormap_path
        )

    def get_color_from_name(self, name: str) -> Tuple[int, int, int, int]:
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
        assert (
            masks.shape[0] == labels.shape[0]
        ), "Number of masks and labels must match"
        assert masks.dtype == torch.bool, "Masks should be boolean arrays"

        # Iterate over each mask and corresponding label
        for mask, label in zip(masks, labels):
            if label.item() in self.id_to_rgba:
                # Retrieve the RGBA color for the label
                r, g, b, _ = self.id_to_rgba[label.item()]

                # Apply the color where the mask is True
                recolored_img[mask] = torch.tensor([r, g, b], dtype=torch.uint8)

        return recolored_img
