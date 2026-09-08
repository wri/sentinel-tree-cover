#!/usr/bin/env python3
import os
import requests
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.io import MemoryFile

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

def download_dem_for_template_tif(
    template_tif: str,
    out_dem_tif: str,
    demtype: str = "COP30",          # e.g., "COP30", "SRTMGL1", etc.
    api_key: str | None = None,      # OpenTopography API key (recommended)
    match_template: bool = True,     # if True: warp to template grid/CRS/shape
    resampling: Resampling = Resampling.bilinear,
    timeout: int = 300,
):
    """
    Download a DEM covering the bounding box of `template_tif` from OpenTopography GlobalDEM API.
    Optionally reproject/resample to exactly match the template tif grid.

    Notes:
      - OpenTopography expects bbox params in WGS84 degrees:
        west/east = lon, south/north = lat
      - API params include demtype, west, south, east, north, outputFormat=GTiff, API_Key.
    """

    if api_key is None:
        api_key = os.environ.get("OPENTOPO_API_KEY") or os.environ.get("OPENTOPO_APIKEY")
    if not api_key:
        raise ValueError(
            "OpenTopography API key missing. Pass api_key=... or set env OPENTOPO_API_KEY."
        )

    # --- Read template bounds and convert to WGS84 lon/lat ---
    with rasterio.open(template_tif) as src:
        template_crs = src.crs
        template_transform = src.transform
        template_width = src.width
        template_height = src.height
        template_profile = src.profile.copy()

        left, bottom, right, top = src.bounds

        # Convert bounds to EPSG:4326 for OT
        if template_crs is None:
            raise ValueError("Template tif has no CRS; cannot compute lon/lat bbox.")
        west, south, east, north = transform_bounds(
            template_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
        )

    # --- Download DEM from OpenTopography ---
    params = {
        "demtype": demtype,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    r = requests.get(OPENTOPO_URL, params=params, stream=True, timeout=timeout)
    r.raise_for_status()

    # If not matching template, just save bytes directly
    if not match_template:
        os.makedirs(os.path.dirname(out_dem_tif) or ".", exist_ok=True)
        with open(out_dem_tif, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return out_dem_tif

    # --- Otherwise: read downloaded GTiff into memory and warp to template grid ---
    dem_bytes = r.content
    with MemoryFile(dem_bytes) as mem:
        with mem.open() as dem:
            # Prepare destination array
            dst = np.zeros((template_height, template_width), dtype=np.float32)

            # Reproject/resample DEM onto template grid
            reproject(
                source=rasterio.band(dem, 1),
                destination=dst,
                src_transform=dem.transform,
                src_crs=dem.crs,
                dst_transform=template_transform,
                dst_crs=template_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

            out_profile = template_profile
            out_profile.update(
                dtype="float32",
                count=1,
                nodata=np.nan,
                compress="deflate",
                predictor=2,
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

            os.makedirs(os.path.dirname(out_dem_tif) or ".", exist_ok=True)
            with rasterio.open(out_dem_tif, "w", **out_profile) as out:
                out.write(dst, 1)

    return out_dem_tif


if __name__ == "__main__":
    # Example:
    # export OPENTOPO_API_KEY="..."
    template = "cos.tif"
    out_dem = "dem.tif"
    download_dem_for_template_tif(template, out_dem, demtype="COP30", match_template=True)
    print("Wrote:", out_dem)
