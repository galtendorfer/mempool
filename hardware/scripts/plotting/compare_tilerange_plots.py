#!/usr/bin/env python3

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import ImageChops
from PIL import Image


DEFAULT_TILE_RANGES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_LAYOUT = "context-pages"
DEFAULT_COLUMNS = 2
DEFAULT_ROWS_PER_PAGE = None
DEFAULT_RASTER_DPI = 120
DEFAULT_OUTPUT_DPI = 120
PAGE_WIDTH_INCHES = 18.6
PAGE_ROW_HEIGHT_INCHES = 6.2
STACKED_PAGE_WIDTH_INCHES = 18.2
STACKED_ROW_HEIGHT_INCHES = 4.9
BOUNDARY_STACKED_PAGE_WIDTH_INCHES = 20.4
BOUNDARY_STACKED_ROW_HEIGHT_INCHES = 5.8
PDFTOPPM_CANDIDATES = [
    "/usr/bin/pdftoppm",
    "/bin/pdftoppm",
    "/usr/local/bin/pdftoppm",
]
PDFTOPPM_PATH = None
SECTION_CONTEXTS = {
    "throughput": [
        ("Throughput", "plots/load_throughput.pdf"),
    ],
    "tile": [
        ("Tile Utilization", "plots/tile/tilerange{tile_range}_tile_ports_utilization.pdf"),
        ("Tile Backpressure", "plots/tile/tilerange{tile_range}_tile_ports_backpressure.pdf"),
    ],
    "subgroup": [
        ("Subgroup Utilization", "plots/subgroup/tilerange{tile_range}_subgroup_ports_utilization.pdf"),
        ("Subgroup Backpressure", "plots/subgroup/tilerange{tile_range}_subgroup_ports_backpressure.pdf"),
    ],
    "group": [
        ("Group Utilization", "plots/group/tilerange{tile_range}_group_ports_utilization.pdf"),
        ("Group Backpressure", "plots/group/tilerange{tile_range}_group_ports_backpressure.pdf"),
    ],
}
DEFAULT_SECTIONS = ["throughput", "tile", "group", "subgroup"]
TOPOLOGY_DEFAULT_SECTIONS = {
    "mempool": ["throughput", "tile", "group"],
    "terapool": ["throughput", "tile", "group", "subgroup"],
}
TOPOLOGY_DEFAULT_TILE_RANGES = {
    "mempool": [1, 2, 4, 8, 16, 32, 64],
    "terapool": [1, 2, 4, 8, 16, 32, 64, 128],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create comparison PDFs for tilerange plots.")
    parser.add_argument("--results-root", default="hardware/results/mempool/latest", help="Root directory containing tilerange result folders.")
    parser.add_argument(
        "--tile-ranges",
        nargs="+",
        type=int,
        default=None,
        help="Tile ranges to include in the comparison grid. Defaults to all tileranges discovered under the results root.",
    )
    parser.add_argument(
        "--output",
        default="hardware/results/mempool/latest/comparisons/latest_tilerange_comparison.pdf",
        help="Output comparison PDF path.",
    )
    parser.add_argument(
        "--layout",
        choices=["context-pages", "single-sheet"],
        default=DEFAULT_LAYOUT,
        help="Comparison layout. 'context-pages' stacks all tile ranges vertically per context; 'single-sheet' keeps the older dense grid.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help="Number of plots to place side by side in the legacy single-sheet layout.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=sorted(SECTION_CONTEXTS.keys()),
        default=None,
        help="Plot sections to include in the comparison PDF. Defaults are topology-specific when omitted.",
    )
    parser.add_argument(
        "--page-per-context",
        action="store_true",
        help="Deprecated alias for --layout context-pages.",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=DEFAULT_ROWS_PER_PAGE,
        help="When using context-pages layout, limit each page to this many tile-range rows. Defaults to all tile ranges on one page.",
    )
    parser.add_argument(
        "--throughput-relative-pdf",
        default="plots/load_throughput_provisional.pdf",
        help="Relative path to the per-run throughput plot PDF used by the throughput section. Defaults to the reconstructed provisional plot when available.",
    )
    parser.add_argument(
        "--raster-dpi",
        type=int,
        default=DEFAULT_RASTER_DPI,
        help="DPI used when rasterizing source plot PDFs with pdftoppm.",
    )
    parser.add_argument(
        "--output-dpi",
        type=int,
        default=DEFAULT_OUTPUT_DPI,
        help="DPI used when writing the final comparison PDF pages.",
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


def infer_topology(results_root: Path) -> Optional[str]:
    normalized_parts = [part.lower() for part in results_root.parts]
    if "terapool" in normalized_parts:
        return "terapool"
    if "mempool" in normalized_parts:
        return "mempool"
    return None


def default_sections_for_results_root(results_root: Path) -> List[str]:
    topology = infer_topology(results_root)
    if topology in TOPOLOGY_DEFAULT_SECTIONS:
        return TOPOLOGY_DEFAULT_SECTIONS[topology]
    return DEFAULT_SECTIONS


def default_tile_ranges_for_results_root(results_root: Path) -> List[int]:
    topology = infer_topology(results_root)
    if topology in TOPOLOGY_DEFAULT_TILE_RANGES:
        return TOPOLOGY_DEFAULT_TILE_RANGES[topology]
    return DEFAULT_TILE_RANGES


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


def discover_tile_ranges(results_root: Path) -> List[int]:
    discovered = set()
    for path in results_root.rglob("tilerange*"):
        name = path.stem if path.is_file() else path.name
        suffix = name.removeprefix("tilerange")
        if suffix.isdigit():
            discovered.add(int(suffix))
            continue
        match = None
        if name.startswith("tilerange"):
            match = name.split("_", 1)[0].removeprefix("tilerange")
        if match and match.isdigit():
            discovered.add(int(match))
    return sorted(discovered)


def resolve_plot_pdf(base_dir: Path, relative_pdf: str, tile_range: int) -> Path:
    formatted_relative = relative_pdf.format(tile_range=tile_range)
    direct_path = base_dir / formatted_relative
    if direct_path.is_file():
        return direct_path

    throughput_fallback_relative = None
    if Path(formatted_relative).name == "load_throughput_provisional.pdf":
        throughput_fallback_relative = Path(formatted_relative).with_name("load_throughput.pdf")
        fallback_direct_path = base_dir / throughput_fallback_relative
        if fallback_direct_path.is_file():
            return fallback_direct_path

    relative_path = Path(formatted_relative)
    if relative_path.parts and relative_path.parts[0] == "plots":
        inner_path = Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else Path(relative_path.name)
        curated_path = base_dir / "plots" / inner_path
        if curated_path.is_file():
            return curated_path

        if throughput_fallback_relative is not None:
            fallback_inner_path = Path(*throughput_fallback_relative.parts[1:]) if len(throughput_fallback_relative.parts) > 1 else Path(throughput_fallback_relative.name)
            fallback_curated_path = base_dir / "plots" / fallback_inner_path
            if fallback_curated_path.is_file():
                return fallback_curated_path

        if relative_path.name.startswith("load_throughput"):
            throughput_path = base_dir / "plots" / "throughput" / f"tilerange{tile_range}_{relative_path.name}"
            if throughput_path.is_file():
                return throughput_path

            if throughput_fallback_relative is not None:
                fallback_throughput_path = base_dir / "plots" / "throughput" / f"tilerange{tile_range}_{throughput_fallback_relative.name}"
                if fallback_throughput_path.is_file():
                    return fallback_throughput_path

    return direct_path


def resolve_pdftoppm() -> str:
    global PDFTOPPM_PATH

    if PDFTOPPM_PATH is not None:
        return PDFTOPPM_PATH

    safe_path_entries = ["/usr/bin", "/bin", "/usr/local/bin"]
    inherited_path = os.environ.get("PATH")
    if inherited_path:
        safe_path_entries.append(inherited_path)
    safe_path = os.pathsep.join(safe_path_entries)

    resolved = shutil.which("pdftoppm", path=safe_path)
    if resolved is None:
        for candidate in PDFTOPPM_CANDIDATES:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                resolved = candidate
                break

    if resolved is None:
        raise SystemExit("pdftoppm not found. Install poppler-utils or pass a valid PATH including pdftoppm.")

    PDFTOPPM_PATH = resolved
    return PDFTOPPM_PATH


def render_pdf_first_page(pdf_path: Path, output_png: Path, raster_dpi: int) -> bool:
    if not pdf_path.is_file():
        return False
    pdftoppm_path = resolve_pdftoppm()
    subprocess_env = os.environ.copy()
    subprocess_env["PATH"] = os.pathsep.join(["/usr/bin", "/bin", "/usr/local/bin", subprocess_env.get("PATH", "")])
    subprocess.run(
        [pdftoppm_path, "-r", str(raster_dpi), "-f", "1", "-singlefile", "-png", str(pdf_path), str(output_png.with_suffix(""))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=subprocess_env,
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


def draw_plot(axis, temp_dir: Path, run_dir: Path, context_title: str, relative_pdf: str, tile_range: int, raster_dpi: int) -> None:
    pdf_path = resolve_plot_pdf(run_dir, relative_pdf, tile_range)
    png_path = temp_dir / f"{context_title.lower().replace(' ', '_')}_tr{tile_range}.png"
    if run_dir.name.startswith("20"):
        title = rf"$\bf{{tilerange{tile_range}}}$  |  run {run_dir.name}"
    else:
        title = rf"$\bf{{tilerange{tile_range}}}$"

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)

    if render_pdf_first_page(pdf_path, png_path, raster_dpi):
        with Image.open(png_path) as image:
            axis.imshow(crop_image_margins(image))
        axis.set_title(title, fontsize=11, loc="left", pad=2)
    else:
        axis.text(0.5, 0.5, f"Missing plot\n{pdf_path.name}", ha="center", va="center", fontsize=10)
        axis.set_title(title, fontsize=11, loc="left", pad=2)


def context_page_dimensions(relative_pdf: str) -> Tuple[float, float]:
    if "throughput" in relative_pdf or "load_throughput" in relative_pdf:
        return STACKED_PAGE_WIDTH_INCHES, STACKED_ROW_HEIGHT_INCHES
    return BOUNDARY_STACKED_PAGE_WIDTH_INCHES, BOUNDARY_STACKED_ROW_HEIGHT_INCHES


def normalize_rows_per_page(rows_per_page: Optional[int], tile_count: int) -> int:
    if rows_per_page is None:
        return max(1, tile_count)
    return max(1, rows_per_page)


def add_single_sheet(output_path: Path, run_map: Dict[int, Path], temp_dir: Path, columns: int, contexts: List[Tuple[str, str]], raster_dpi: int, output_dpi: int) -> None:
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
            draw_plot(axis, temp_dir, run_dir, context_title, relative_pdf, tile_range, raster_dpi)

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
    fig.savefig(output_path, dpi=output_dpi)
    plt.close(fig)


def add_context_pages(
    output_path: Path,
    run_map: Dict[int, Path],
    temp_dir: Path,
    rows_per_page: int,
    contexts: List[Tuple[str, str]],
    raster_dpi: int,
    output_dpi: int,
) -> None:
    ordered_tile_ranges = list(run_map.keys())
    page_rows = normalize_rows_per_page(rows_per_page, len(ordered_tile_ranges))

    with PdfPages(output_path) as pdf:
        for context_title, relative_pdf in contexts:
            total_pages = math.ceil(len(ordered_tile_ranges) / page_rows)
            for page_index in range(total_pages):
                page_tile_ranges = ordered_tile_ranges[page_index * page_rows : (page_index + 1) * page_rows]
                current_rows = len(page_tile_ranges)
                page_width_inches, row_height_inches = context_page_dimensions(relative_pdf)
                fig, axes = plt.subplots(
                    current_rows,
                    1,
                    figsize=(page_width_inches, current_rows * row_height_inches),
                )
                if hasattr(axes, "flat"):
                    axes_list = list(axes.flat)
                else:
                    axes_list = [axes]

                for axis, tile_range in zip(axes_list, page_tile_ranges):
                    run_dir = run_map[tile_range]
                    draw_plot(axis, temp_dir, run_dir, context_title, relative_pdf, tile_range, raster_dpi)

                if total_pages > 1:
                    title = f"{context_title}  |  page {page_index + 1}/{total_pages}"
                else:
                    title = context_title
                fig.suptitle(title, fontsize=20, fontweight="bold", y=0.995)
                fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.982), h_pad=1.1)
                pdf.savefig(fig, dpi=output_dpi)
                plt.close(fig)


def main() -> None:
    args = parse_args()
    layout = "context-pages" if args.page_per_context else args.layout
    results_root = Path(args.results_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = args.sections or default_sections_for_results_root(results_root)
    contexts = get_contexts(sections, args.throughput_relative_pdf)
    tile_ranges = args.tile_ranges or default_tile_ranges_for_results_root(results_root)

    run_map = {}
    for tile_range in tile_ranges:
        run_dir = latest_run_dir(results_root, tile_range)
        if run_dir is not None:
            run_map[tile_range] = run_dir
        elif resolve_plot_pdf(results_root, args.throughput_relative_pdf, tile_range).is_file() or (results_root / "plots").is_dir():
            run_map[tile_range] = results_root

    if not run_map:
        raise SystemExit("No tilerange result directories found.")

    if args.columns < 1:
        raise SystemExit("--columns must be at least 1")
    if args.rows_per_page is not None and args.rows_per_page < 1:
        raise SystemExit("--rows-per-page must be at least 1")
    if args.raster_dpi < 50:
        raise SystemExit("--raster-dpi must be at least 50")
    if args.output_dpi < 50:
        raise SystemExit("--output-dpi must be at least 50")

    with tempfile.TemporaryDirectory(prefix="compare_tilerange_plots_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if layout == "context-pages":
            add_context_pages(output_path, run_map, temp_dir, args.rows_per_page, contexts, args.raster_dpi, args.output_dpi)
        else:
            add_single_sheet(output_path, run_map, temp_dir, args.columns, contexts, args.raster_dpi, args.output_dpi)

    print(output_path)


if __name__ == "__main__":
    main()