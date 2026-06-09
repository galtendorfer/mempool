"""Shared output path helpers for trace-analysis plot scripts."""

from __future__ import annotations

from pathlib import Path


def figure_path(output_base: Path, extension: str) -> Path:
    """Return the output path for a figure extension.

    PNG files stay next to the visible plot set. PDF files are stored in a
    sibling ``pdf/`` folder to keep plot directories easy to scan.
    """
    suffix = extension.lstrip(".")
    if suffix == "pdf":
        return output_base.parent / "pdf" / f"{output_base.name}.pdf"
    return output_base.with_suffix(f".{suffix}")


def pdf_path_for_png(png_path: Path) -> Path:
    return png_path.parent / "pdf" / f"{png_path.stem}.pdf"


def data_path(output_dir: Path, filename: str) -> Path:
    return output_dir / "data" / filename