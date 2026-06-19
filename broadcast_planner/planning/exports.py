"""Export maps, tables, and PDF reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from broadcast_planner.core.crs import ANALYSIS_CRS, gdf_to_display
from broadcast_planner.core.grids import GridSpec
from broadcast_planner.core.sites import SiteCollection
from broadcast_planner.dvb_t.sfn import sfn_delay_table


def export_geotiff(grid: GridSpec, array: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.write_geotiff(path, array.astype(np.float32))
    return path


def export_sites_geojson(sites: SiteCollection, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = sites.to_geodataframe()
    gdf_display = gdf_to_display(gdf)
    gdf_display.to_file(path, driver="GeoJSON")
    return path


def export_kpi_csv(stats: Dict, path: Path, extra_rows: Optional[pd.DataFrame] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([stats])
    if extra_rows is not None:
        df = pd.concat([df, extra_rows], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def export_sfn_table_csv(sites: SiteCollection, path: Path) -> Path:
    df = pd.DataFrame(sfn_delay_table(sites))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_coverage_png(array: np.ndarray, path: Path, title: str = "Coverage") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(array, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_pdf_report(
    output_path: Path,
    title: str,
    population_stats: Dict,
    sites: SiteCollection,
    map_paths: Dict[str, Path],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, title)
    y -= 1.5 * cm
    c.setFont("Helvetica", 11)
    for key, val in population_stats.items():
        c.drawString(2 * cm, y, f"{key}: {val}")
        y -= 0.6 * cm
    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Sites")
    y -= 0.8 * cm
    c.setFont("Helvetica", 10)
    for site in sites.sites:
        c.drawString(
            2 * cm,
            y,
            f"{site.id}: ERP={site.erp_kw} kW, h={site.antenna_height_m} m, {site.topology}",
        )
        y -= 0.5 * cm
    for label, img_path in map_paths.items():
        if y < 8 * cm:
            c.showPage()
            y = height - 2 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, label)
        y -= 0.5 * cm
        c.drawImage(str(img_path), 2 * cm, y - 8 * cm, width=14 * cm, height=8 * cm)
        y -= 9 * cm
    c.save()
    return output_path
