"""Shared discrete colour scales for source-port fan-in heatmaps."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap, LinearSegmentedColormap, ListedColormap


AVERAGE_STEP = 0.25


def _discrete_cmap(colors: str | Sequence[str], levels: int) -> Colormap:
    if isinstance(colors, str):
        return plt.get_cmap(colors, levels)
    palette = list(colors)
    if levels <= len(palette):
        return ListedColormap(palette[:levels])
    return LinearSegmentedColormap.from_list("fanin_quantized", palette, N=levels)


def _format_level(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fanin_cmap_norm(max_cores: int, colors: str | Sequence[str], averaged: bool) -> tuple[Colormap, BoundaryNorm]:
    """Build a discrete 0..max_cores fan-in colour scale.

    Exact cycle data uses integer-centred bins. Window-averaged data uses
    quarter-step bins, equivalent to rounding each average to the nearest 0.25.
    """
    if averaged:
        levels = int(round(max_cores / AVERAGE_STEP)) + 1
        cmap = _discrete_cmap(colors, levels)
        boundaries = np.array([
            level * AVERAGE_STEP - AVERAGE_STEP / 2.0
            for level in range(levels + 1)
        ])
    else:
        cmap = _discrete_cmap(colors, max_cores + 1)
        boundaries = np.arange(-0.5, max_cores + 1.5, 1)
    return cmap, BoundaryNorm(boundaries, cmap.N, clip=True)


def fanin_colorbar_ticks(max_cores: int, averaged: bool) -> tuple[list[float], list[str]]:
    if not averaged:
        values = list(range(max_cores + 1))
        return [float(value) for value in values], [str(value) for value in values]
    levels = int(round(max_cores / AVERAGE_STEP)) + 1
    ticks = [level * AVERAGE_STEP for level in range(levels)]
    labels = [_format_level(tick) for tick in ticks]
    return ticks, labels


def add_fanin_colorbar(fig, image, axes, max_cores: int, averaged: bool, label: str, **kwargs):
    ticks, labels = fanin_colorbar_ticks(max_cores, averaged)
    colorbar = fig.colorbar(image, ax=axes, ticks=ticks, **kwargs)
    colorbar.set_ticklabels(labels)
    colorbar.set_label(label)
    return colorbar