#!/usr/bin/env python3

import argparse
import math
import subprocess
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import ImageChops
from PIL import Image


DEFAULT_TILE_RANGES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_COLUMNS = 2
DEFAULT_ROWS_PER_PAGE = 3
RASTER_DPI = 400
OUTPUT_DPI = 400
PAGE_WIDTH_INCHES = 18
PAGE_ROW_HEIGHT_INCHES = 5.8
SECTION_CONTEXTS = {
    "throughput": [
        ("Throughput", "plots/load_throughput.pdf"),
    ],
    "tile": [
        ("Tile Utilization", "plots/tile/tilerange{tile_range}_tile_ports_utilization.pdf"),
        ("Tile Backpressure", "plots/tile/tilerange{tile_range}_tile_ports_backpressure.pdf"),
        ("Tile Split Utilization", "plots/tile/tilerange{tile_range}_tile_split_utilization.pdf"),
    ],
    "subgroup": [
        ("Subgroup Utilization", "plots/subgroup/tilerange{tile_range}_subgroup_ports_utilization.pdf"),
        ("Subgroup Backpressure", "plots/subgroup/tilerange{tile_range}_subgroup_ports_backpressure.pdf"),
        ("Subgroup Split Utilization", "plots/subgroup/tilerange{tile_range}_subgroup_split_utilization.pdf"),
    ],
    "group": [
        ("Group Utilization", "plots/group/tilerange{tile_range}_group_ports_utilization.pdf"),
        ("Group Backpressure", "plots/group/tilerange{tile_range}_group_ports_backpressure.pdf"),
        ("Group Split Utilization", "plots/group/tilerange{tile_range}_group_split_utilization.pdf"),
    ],
}
DEFAULT_SECTIONS = ["throughput", "tile", "group"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a single-sheet comparison PDF for tilerange plots.")
    parser.add_argument("--results-root", default="hardware/results/mempool/latest", help="Root directory containing tilerange result folders.")
    parser.add_argument(
        "--tile-ranges",
        nargs="+",
        type=int,
        default=DEFAULT_TILE_RANGES,
        help="Tile ranges to include in the comparison grid.",
    )
    parser.add_argument(
        "--output",
        default="hardware/results/mempool/latest/comparisons/latest_tilerange_comparison.pdf",
        help="Output comparison PDF path.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help="Number of plots to place side by side within each context section.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=sorted(SECTION_CONTEXTS.keys()),
        default=DEFAULT_SECTIONS,
        help="Plot sections to include in the comparison PDF.",
    )
    parser.add_argument(
        "--page-per-context",
        action="store_true",
        help="Render one PDF page per context so each metric gets a larger layout.",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=DEFAULT_ROWS_PER_PAGE,
        help="When using --page-per-context, limit each page to this many plot rows before continuing on a new page.",
    )
    parser.add_argument(
        "--throughput-relative-pdf",
        default="plots/load_throughput.pdf",
        help="Relative path to the per-run throughput plot PDF used by the throughput section.",
    )
    return parser.parse_args()


def get_contexts(sections: List[str], throughput_relative_pdf: str) -> List[Tuple[str, str]]:
    contexts = []
    for section in sections:
        if section == "throughput":
            contexts.append(("Throughput", throughput_relative_pdf))
        else:
            contexts.extend(SECTION_CONTEXTS[section])
    return contexts


def tilerange_dir_for_root(results_root: Path, tile_range: int) -> Optional[Path]:
    candidates = [
        results_root / f"tilerange{tile_range}",
        results_root / "latest" / f"tilerange{tile_range}",
        results_root / "runs" / f"tilerange{tile_range}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def latest_run_dir(results_root: Path, tile_range: int) -> Optional[Path]:
    tilerange_dir = tilerange_dir_for_root(results_root, tile_range)
    if tilerange_dir is None:
        return None

    if (tilerange_dir / "plots").is_dir() or (tilerange_dir / "data").is_dir() or (tilerange_dir / "summary").is_dir():
        return tilerange_dir

    if not tilerange_dir.is_dir():
        return None
    run_dirs = sorted(path for path in tilerange_dir.iterdir() if path.is_dir())
    return run_dirs[-1] if run_dirs else None


def render_pdf_first_page(pdf_path: Path, output_png: Path) -> bool:
    if not pdf_path.is_file():
        return False
    subprocess.run(
        ["pdftoppm", "-r", str(RASTER_DPI), "-f", "1", "-singlefile", "-png", str(pdf_path), str(output_png.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_png.is_file()


def crop_image_margins(image: Image.Image) -> Image.Image:
    rgb_image = image.convert("RGB")
    white_background = Image.new("RGB", rgb_image.size, (255, 255, 255))
    diff = ImageChops.difference(rgb_image, white_background)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb_image

    left, top, right, bottom = bbox
    pad = 8
    left = max(left - pad, 0)
    top = max(top - pad, 0)
    right = min(right + pad, rgb_image.width)
    bottom = min(bottom + pad, rgb_image.height)
    return rgb_image.crop((left, top, right, bottom))


def draw_plot(axis, temp_dir: Path, run_dir: Path, context_title: str, relative_pdf: str, tile_range: int) -> None:
    pdf_path = run_dir / relative_pdf.format(tile_range=tile_range)
    png_path = temp_dir / f"{context_title.lower().replace(' ', '_')}_tr{tile_range}.png"
    title = rf"$\bf{{tilerange{tile_range}}}$  |  run {run_dir.name}"

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)

    if render_pdf_first_page(pdf_path, png_path):
        with Image.open(png_path) as image:
            axis.imshow(crop_image_margins(image))
        axis.set_title(title, fontsize=11, loc="left", pad=2)
    else:
        axis.text(0.5, 0.5, f"Missing plot\n{pdf_path.name}", ha="center", va="center", fontsize=10)
        axis.set_title(title, fontsize=11, loc="left", pad=2)


def add_single_sheet(output_path: Path, run_map: Dict[int, Path], temp_dir: Path, columns: int, contexts: List[Tuple[str, str]]) -> None:
    ordered_tile_ranges = list(run_map.keys())
    context_columns = columns
    context_rows = math.ceil(len(contexts) / context_columns)
    row_count = context_rows * (1 + len(ordered_tile_ranges))
    height_ratios = []
    for _ in range(context_rows):
        height_ratios.append(0.22)
        height_ratios.extend([4.2] * len(ordered_tile_ranges))

    fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, row_count * 1.8))
    grid = fig.add_gridspec(row_count, context_columns, height_ratios=height_ratios)

    for context_index, (context_title, relative_pdf) in enumerate(contexts):
        context_row_group = context_index // context_columns
        context_column = context_index % context_columns
        base_row = context_row_group * (1 + len(ordered_tile_ranges))

        header_axis = fig.add_subplot(grid[base_row, context_column])
        header_axis.axis("off")
        header_axis.text(0.0, 0.5, context_title, fontsize=16, fontweight="bold", ha="left", va="center")

        for tile_offset, tile_range in enumerate(ordered_tile_ranges, start=1):
            axis = fig.add_subplot(grid[base_row + tile_offset, context_column])
            run_dir = run_map[tile_range]
            draw_plot(axis, temp_dir, run_dir, context_title, relative_pdf, tile_range)

    unused_context_slots = context_rows * context_columns - len(contexts)
    if unused_context_slots > 0:
        start_index = len(contexts)
        for blank_index in range(start_index, start_index + unused_context_slots):
            context_row_group = blank_index // context_columns
            context_column = blank_index % context_columns
            base_row = context_row_group * (1 + len(ordered_tile_ranges))
            for row in range(base_row, base_row + 1 + len(ordered_tile_ranges)):
                axis = fig.add_subplot(grid[row, context_column])
                axis.axis("off")

    plt.tight_layout(rect=(0.005, 0.003, 0.995, 0.997), h_pad=0.7, w_pad=0.8)
    fig.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(fig)


def add_context_pages(
    output_path: Path,
    run_map: Dict[int, Path],
    temp_dir: Path,
    columns: int,
    rows_per_page: int,
    contexts: List[Tuple[str, str]],
) -> None:
    ordered_tile_ranges = list(run_map.keys())
    page_columns = max(1, columns)
    page_rows = max(1, rows_per_page)
    plots_per_page = page_columns * page_rows

    with PdfPages(output_path) as pdf:
        for context_title, relative_pdf in contexts:
            total_pages = math.ceil(len(ordered_tile_ranges) / plots_per_page)
            for page_index in range(total_pages):
                page_tile_ranges = ordered_tile_ranges[page_index * plots_per_page : (page_index + 1) * plots_per_page]
                fig, axes = plt.subplots(
                    page_rows,
                    page_columns,
                    figsize=(PAGE_WIDTH_INCHES, page_rows * PAGE_ROW_HEIGHT_INCHES),
                )
                if hasattr(axes, "flat"):
                    axes_list = list(axes.flat)
                else:
                    axes_list = [axes]

                for axis, tile_range in zip(axes_list, page_tile_ranges):
                    run_dir = run_map[tile_range]
                    draw_plot(axis, temp_dir, run_dir, context_title, relative_pdf, tile_range)

                for axis in axes_list[len(page_tile_ranges):]:
                    axis.axis("off")

                if total_pages > 1:
                    title = f"{context_title}  |  page {page_index + 1}/{total_pages}"
                else:
                    title = context_title
                fig.suptitle(title, fontsize=20, fontweight="bold", y=0.995)
                fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.975), h_pad=1.0, w_pad=0.8)
                pdf.savefig(fig, dpi=OUTPUT_DPI)
                plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contexts = get_contexts(args.sections, args.throughput_relative_pdf)

    run_map = {}
    for tile_range in args.tile_ranges:
        run_dir = latest_run_dir(results_root, tile_range)
        if run_dir is not None:
            run_map[tile_range] = run_dir

    if not run_map:
        raise SystemExit("No tilerange result directories found.")

    if args.columns < 1:
        raise SystemExit("--columns must be at least 1")
    if args.rows_per_page < 1:
        raise SystemExit("--rows-per-page must be at least 1")

    with tempfile.TemporaryDirectory(prefix="compare_tilerange_plots_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if args.page_per_context:
            add_context_pages(output_path, run_map, temp_dir, args.columns, args.rows_per_page, contexts)
        else:
            add_single_sheet(output_path, run_map, temp_dir, args.columns, contexts)

    print(output_path)


if __name__ == "__main__":
    main()