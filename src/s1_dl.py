#!/usr/bin/env python3
"""
Planetary Computer Sentinel-1 RTC quarterly composites (VV/VH) -> HKL.

Emulates your existing outputs:
  /{YEAR}/raw/{X_tile}/{Y_tile}/raw/misc/s1_dates_{X_tile}X{Y_tile}Y.hkl
  /{YEAR}/raw/{X_tile}/{Y_tile}/raw/s1/{X_tile}X{Y_tile}Y.hkl

Produces:
  raw/s1/{X}X{Y}Y.hkl           (uint16, (12,H,W,2))  # 4 quarters x 3 repeats = 12
  raw/misc/s1_dates_{tile}.hkl  (int64, (12,))        # [45,45,45,135,135,135,225,225,225,315,315,315]

Key differences vs your GRD pipeline:
  - No DEM, no XML calibration/noise, no terrain flattening: RTC is already corrected.
  - Uses Planetary Computer STAC + signed assets.
"""

import os
import sys
import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.features import bounds as featureBounds

from pystac_client import Client
import planetary_computer as pc
from shapely.geometry import shape, box

import hickle as hkl
from loguru import logger

import obstore as obs
from obstore.store import LocalStore, from_url


# -------------------- Constants --------------------
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
RTC_COLLECTION = "sentinel-1-rtc"
ASSETS_PREF = ["vv", "vh"]  # desired order


# -------------------- Small utils --------------------
def _elapsed_ms(t_start: float) -> float:
    return (time.perf_counter() - t_start) * 1000.0

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Expand a point bbox by ~degrees (same style as your code)."""
    multiplier = 1 / 360
    bbx = initial_bbx.copy()
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx

def bbox2geojson(bbox: list) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]]]
    }

def coverage_fraction(item, tile_bounds: tuple) -> float:
    """Compute fraction of tile area covered by the item's footprint."""
    try:
        tile_poly = box(*tile_bounds)
        item_poly = shape(item.geometry)
        inter = tile_poly.intersection(item_poly)
        if tile_poly.is_empty or tile_poly.area == 0:
            return 0.0
        return float(inter.area / tile_poly.area)
    except Exception as e:
        logger.warning(f"coverage_fraction failed for {getattr(item, 'id', 'unknown')}: {e}")
        return 0.0

def obstore_put_hkl(store, relpath: str, obj) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".hkl", delete=False)
    tmp.close()
    try:
        hkl.dump(obj, tmp.name, mode="w", compression="gzip")
        with open(tmp.name, "rb") as f:
            obs.put(store, relpath, f.read())
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def obstore_put_text(store, relpath: str, text: str) -> None:
    try:
        obs.put(store, relpath, text.encode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to write text sidecar {relpath}: {e}")

def compute_band_stats(arr: np.ndarray) -> dict:
    vals = arr.astype(np.float32)
    total = int(vals.size)
    mask = vals > 0
    valid = vals[mask]
    if valid.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "std": 0.0,
                "p5": 0.0, "p50": 0.0, "p95": 0.0,
                "valid_ratio": 0.0, "count": total, "valid_count": 0}
    p5, p50, p95 = np.percentile(valid, [5, 50, 95])
    return {"min": int(valid.min()), "max": int(valid.max()),
            "mean": float(valid.mean()), "std": float(valid.std()),
            "p5": float(p5), "p50": float(p50), "p95": float(p95),
            "valid_ratio": float(valid.size / total) if total > 0 else 0.0,
            "count": total, "valid_count": int(valid.size)}

def _orbit_query(orbit_direction: str) -> dict:
    """Planetary Computer uses sat:orbit_state (ASCENDING/DESCENDING) commonly."""
    od = (orbit_direction or "BOTH").upper()
    if od == "ASCENDING":
        return {"sat:orbit_state": {"eq": "ascending"}}
    if od == "DESCENDING":
        return {"sat:orbit_state": {"eq": "descending"}}
    return {}

def _quarter_windows(year: int):
    # Same quarter “center-ish” windows you used
    return {
        "Q1": (f"{year}-01-15", f"{year}-03-15"),
        "Q2": (f"{year}-04-15", f"{year}-06-15"),
        "Q3": (f"{year}-07-15", f"{year}-09-15"),
        "Q4": (f"{year}-10-15", f"{year}-12-15"),
    }

def get_quarterly_scenes_by_coverage(items: list, year: int, tile_bounds: tuple,
                                     coverage_threshold: float = 0.95,
                                     k_scenes: int = 3) -> tuple:
    quarters = _quarter_windows(year)
    selected = {}
    quarter_meta = []

    for q_name, (start, end) in quarters.items():
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)

        quarter_items = []
        for item in items:
            # PC pystac items typically have item.datetime populated; fallback to properties datetime
            dt = item.datetime.isoformat() if item.datetime else item.properties.get("datetime")
            if not dt:
                continue
            item_dt = np.datetime64(dt)
            if start_dt <= item_dt <= end_dt:
                quarter_items.append(item)

        if not quarter_items:
            logger.warning(f"{q_name}: No scenes found")
            selected[q_name] = []
            quarter_meta.append({
                "quarter": q_name, "candidate_count": 0, "selected_ids": [],
                "selected_datetimes": [], "selected_orbit": None,
                "coverage_fractions": [], "k_selected": 0
            })
            continue

        scored = []
        for it in quarter_items:
            cf = coverage_fraction(it, tile_bounds)
            scored.append((cf, it))
        scored.sort(key=lambda t: t[0], reverse=True)

        top_k = []
        for cf, it in scored[:k_scenes]:
            if cf >= coverage_threshold or len(top_k) == 0:
                top_k.append((cf, it))
            else:
                break
        if not top_k:
            top_k = [scored[0]]

        selected[q_name] = [it for cf, it in top_k]
        coverages = [cf for cf, it in top_k]

        quarter_meta.append({
            "quarter": q_name,
            "candidate_count": len(scored),
            "selected_ids": [it.id for cf, it in top_k],
            "selected_datetimes": [
                (it.datetime.isoformat() if it.datetime else it.properties.get("datetime"))
                for cf, it in top_k
            ],
            "selected_orbit": top_k[0][1].properties.get("sat:orbit_state") if top_k else None,
            "coverage_fractions": coverages,
            "k_selected": len(top_k),
        })

        logger.info(f"{q_name}: Selected {len(top_k)} scene(s) coverages={[f'{c:.3f}' for c in coverages]}")

    return selected, quarter_meta

def composite_quarter_scenes(scene_arrays: list, method: str = "median") -> np.ndarray:
    """
    scene_arrays: list of (bands,H,W) uint16 arrays
    returns: (bands,H,W) uint16
    """
    if not scene_arrays:
        raise ValueError("No scenes to composite")
    if len(scene_arrays) == 1:
        return scene_arrays[0]
    stacked = np.stack(scene_arrays, axis=0).astype(np.float32)  # (k,b,h,w)
    stacked = np.where(stacked == 0, np.nan, stacked)
    with np.errstate(all="ignore"):
        if method == "median":
            out = np.nanmedian(stacked, axis=0)
        elif method == "mean":
            out = np.nanmean(stacked, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    out = np.where(np.isnan(out), 0, out)
    return out.astype(np.uint16)

def _detect_vv_vh_assets(item) -> list:
    """Return asset keys in preferred order (vv,vh) if present; else best-effort detection."""
    keys = list(item.assets.keys())
    lower = {k.lower(): k for k in keys}

    found = []
    for want in ASSETS_PREF:
        if want in item.assets:
            found.append(want)
        elif want in lower:
            found.append(lower[want])

    # fallback: substring match
    if len(found) < 2:
        for k in keys:
            kl = k.lower()
            if "vv" in kl and all("vv" not in f.lower() for f in found):
                found.append(k)
            if "vh" in kl and all("vh" not in f.lower() for f in found):
                found.append(k)

    # keep order vv then vh if both exist
    def _rank(k):
        kl = k.lower()
        return 0 if "vv" in kl else (1 if "vh" in kl else 2)
    found = sorted(set(found), key=_rank)

    # keep only first two (vv,vh) if we got extras
    if len(found) > 2:
        found = found[:2]
    return found

def read_rtc_band_to_tile_uint16(href: str, bounds_lonlat: tuple, target_crs: str = "EPSG:4326") -> np.ndarray:
    """
    Read RTC COG over the tile bounds, warped to EPSG:4326, return uint16 scaled.
    Scaling: clip [0,1] -> [0,65535], matching your OPTION A style.
    """
    # Avoid GDAL reading directory listings for cloud paths
    env = rio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

    with env:
        with rio.open(href) as src:
            with WarpedVRT(
                src,
                crs=target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0,
            ) as vrt:
                win = from_bounds(*bounds_lonlat, transform=vrt.transform)
                win = win.round_offsets().round_lengths()
                arr = vrt.read(1, window=win).astype(np.float32)

    # Replace nodata/negatives with 0 (RTC can have 0 nodata)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)

    # Your scaling OPTION A
    scaled = np.clip(arr, 0.0, 1.0) * 65535.0
    return scaled.astype(np.uint16)


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Planetary Computer S1 RTC quarterly composites -> HKL")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--X_tile", type=int, required=True)
    ap.add_argument("--Y_tile", type=int, required=True)
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--expansion", type=int, default=300)
    ap.add_argument("--coverage-threshold", type=float, default=0.95)
    ap.add_argument("--orbit-direction", type=str, choices=["ASCENDING", "DESCENDING", "BOTH"], default="ASCENDING")
    ap.add_argument("--k-scenes", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--run-metadata", action="store_true")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    t_all = time.perf_counter()

    # Output store (local or s3)
    if args.dest.startswith("s3://"):
        store = from_url(args.dest, region="us-east-1")
    else:
        os.makedirs(args.dest, exist_ok=True)
        store = LocalStore(prefix=args.dest)
    base_key = f"{args.year}/{args.X_tile}/{args.Y_tile}/raw"
    s1_dir_key = f"{base_key}/s1"
    misc_key = f"{base_key}/misc"
    fn_s1_key = f"{s1_dir_key}/{args.X_tile}X{args.Y_tile}Y.hkl"
    fn_s1_dates_key = f"{misc_key}/s1_dates_{args.X_tile}X{args.Y_tile}Y.hkl"

    # Build bbox
    initial_bbx = [args.lon, args.lat, args.lon, args.lat]
    bbx = make_bbox(initial_bbx, expansion=args.expansion / 30)
    tile_bounds = featureBounds(bbox2geojson(bbx))  # (minx,miny,maxx,maxy) in lon/lat
    print(tile_bounds)
    # STAC query (Planetary Computer)
    client = Client.open(PC_STAC)
    stac_query = _orbit_query(args.orbit_direction)

    # Note: use intersects polygon like your script
    search = client.search(
        collections=[RTC_COLLECTION],
        datetime=f"{args.year}-01-01/{args.year}-12-31",
        intersects=bbox2geojson(bbx),
        query=stac_query,
        limit=500,
    )

    logger.debug(f"STAC search: {search.url_with_parameters()}")
    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-1 RTC items found for query.")

    logger.info(f"Found {len(items)} PC RTC scenes for {args.year} ({args.orbit_direction})")

    # Sign items so asset hrefs are readable (SAS tokens)
    items = [pc.sign(it) for it in items]

    # Select top-K per quarter by coverage
    quarterly_items, quarter_meta = get_quarterly_scenes_by_coverage(
        items, args.year, tile_bounds, args.coverage_threshold, args.k_scenes
    )

    # Determine bands/asset keys from available items (prefer vv,vh)
    bands = None
    for q, lst in quarterly_items.items():
        if lst:
            bands = _detect_vv_vh_assets(lst[0])
            break
    if not bands or len(bands) == 0:
        bands = ["vv"]
        logger.warning("Could not detect vv/vh assets; falling back to vv only")
    else:
        # ensure vv then vh ordering
        bands = sorted(bands, key=lambda k: 0 if "vv" in k.lower() else (1 if "vh" in k.lower() else 2))
    logger.info(f"Using asset keys: {bands}")

    bounds = tile_bounds
    target_crs = "EPSG:4326"

    quarterly_arrays = []
    quarter_band_stats = {q: {} for q in ["Q1", "Q2", "Q3", "Q4"]}

    def process_single_item(item):
        # returns (bands,H,W)
        band_arrays = []
        for b in bands:
            if b not in item.assets:
                logger.warning(f"Asset '{b}' missing in {item.id}; writing zeros for that band")
                band_arrays.append(None)
                continue
            href = item.assets[b].href
            arr_u16 = read_rtc_band_to_tile_uint16(href, bounds_lonlat=bounds, target_crs=target_crs)
            band_arrays.append(arr_u16)

        # Fill missing bands with zeros matching first band shape
        shape = None
        for a in band_arrays:
            if a is not None:
                shape = a.shape
                break
        if shape is None:
            shape = (512, 512)

        fixed = []
        for a in band_arrays:
            if a is None:
                fixed.append(np.zeros(shape, dtype=np.uint16))
            else:
                fixed.append(a)
        return np.stack(fixed, axis=0)  # (bands,H,W)

    # Process each quarter
    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        item_list = quarterly_items.get(q_name, [])
        if not item_list:
            logger.warning(f"No data for {q_name}, using zeros")
            # size fallback
            if quarterly_arrays:
                h, w = quarterly_arrays[0].shape[1:]
            else:
                h, w = 512, 512
            quarter_data = np.zeros((len(bands), h, w), dtype=np.uint16)
        else:
            logger.info(f"Processing {q_name}: {len(item_list)} scene(s)")
            scene_arrays = []
            for idx, item in enumerate(item_list):
                logger.info(f"  Scene {idx+1}/{len(item_list)}: {item.id}")
                scene = process_single_item(item)
                scene_arrays.append(scene)
                logger.debug(f"    scene shape={scene.shape} range={scene.min()}..{scene.max()}")

            # Composite scenes within quarter (median)
            if len(scene_arrays) > 1:
                # pad to max shape if needed (rare but keep parity with your code)
                max_h = max(a.shape[1] for a in scene_arrays)
                max_w = max(a.shape[2] for a in scene_arrays)
                padded = []
                for a in scene_arrays:
                    if a.shape[1] < max_h or a.shape[2] < max_w:
                        pad_h = max_h - a.shape[1]
                        pad_w = max_w - a.shape[2]
                        a = np.pad(a, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
                    padded.append(a)
                quarter_data = composite_quarter_scenes(padded, method="median")
            else:
                quarter_data = scene_arrays[0]

        # Stats
        try:
            for bi, bname in enumerate(bands):
                quarter_band_stats[q_name][bname] = compute_band_stats(quarter_data[bi])
        except Exception as e:
            logger.warning(f"Failed computing band stats for {q_name}: {e}")

        quarterly_arrays.append(quarter_data)

    # Ensure all quarters same shape
    max_bands = max(a.shape[0] for a in quarterly_arrays)
    max_h = max(a.shape[1] for a in quarterly_arrays)
    max_w = max(a.shape[2] for a in quarterly_arrays)

    def _pad(arr):
        pad_b = max_bands - arr.shape[0]
        pad_h = max_h - arr.shape[1]
        pad_w = max_w - arr.shape[2]
        if pad_b > 0:
            arr = np.pad(arr, ((0, pad_b), (0, 0), (0, 0)), mode="edge")
        if pad_h > 0 or pad_w > 0:
            mode = "reflect" if (arr.shape[1] > 1 and arr.shape[2] > 1) else "edge"
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
        return arr

    quarterly_arrays = [_pad(a) for a in quarterly_arrays]

    # Stack (4,b,h,w) -> repeat each quarter 3 times -> (12,b,h,w) -> transpose to (12,h,w,b)
    all_quarters = np.stack(quarterly_arrays, axis=0)          # (4,B,H,W)
    repeated = np.repeat(all_quarters, 3, axis=0)              # (12,B,H,W)
    final_array = np.transpose(repeated, (0, 2, 3, 1))         # (12,H,W,B)

    # Dates (same as your code)
    quarter_days = [45, 135, 225, 315]
    s1_dates = np.repeat(quarter_days, 3).astype(np.int64)

    # Write outputs
    t_write = time.perf_counter()
    obstore_put_hkl(store, fn_s1_key, final_array)
    obstore_put_hkl(store, fn_s1_dates_key, s1_dates)

    full_s1_path = f"{args.dest.rstrip('/')}/{fn_s1_key}"
    full_dates_path = f"{args.dest.rstrip('/')}/{fn_s1_dates_key}"

    logger.success(f"S1 RTC quarterly composites saved: {full_s1_path}")
    logger.success(f"Shape: {final_array.shape}, dtype: {final_array.dtype}")
    logger.success(f"Range: {int(final_array.min())} to {int(final_array.max())}")
    logger.success(f"S1 quarterly dates saved: {full_dates_path}")
    logger.info(f"Write {_elapsed_ms(t_write):.0f} ms; total {_elapsed_ms(t_all):.0f} ms")

    # Optional metadata sidecar
    if args.run_metadata:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_key = f"{misc_key}/s1_meta_{args.X_tile}X{args.Y_tile}Y_{ts}.txt"
            lines = []
            lines.append("Planetary Computer Sentinel-1 RTC quarterly processing metadata")
            lines.append(f"timestamp: {ts}")
            lines.append(f"year: {args.year}")
            lines.append(f"lon: {args.lon}")
            lines.append(f"lat: {args.lat}")
            lines.append(f"X_tile: {args.X_tile}")
            lines.append(f"Y_tile: {args.Y_tile}")
            lines.append(f"bbox: {bbx}")
            lines.append(f"bounds: {bounds}")
            lines.append(f"items_found: {len(items)}")
            lines.append(f"orbit_direction: {args.orbit_direction}")
            lines.append(f"coverage_threshold: {args.coverage_threshold}")
            lines.append(f"k_scenes: {args.k_scenes}")
            lines.append(f"assets: {bands}")
            lines.append("quarters:")
            for qm in quarter_meta:
                lines.append(f"  - quarter: {qm['quarter']}")
                lines.append(f"    k_selected: {qm['k_selected']}")
                lines.append(f"    selected_ids: {qm['selected_ids']}")
                lines.append(f"    selected_datetimes: {qm['selected_datetimes']}")
                lines.append(f"    orbit: {qm['selected_orbit']}")
                lines.append(f"    coverage_fractions: {qm['coverage_fractions']}")
                lines.append(f"    candidate_count: {qm['candidate_count']}")
            lines.append(f"output_shape: {list(final_array.shape)}")
            lines.append(f"output_dtype: {str(final_array.dtype)}")
            lines.append(f"output_min: {int(final_array.min())}")
            lines.append(f"output_max: {int(final_array.max())}")
            # per-quarter stats
            lines.append("quarter_band_stats:")
            for qn in ["Q1","Q2","Q3","Q4"]:
                lines.append(f"  {qn}:")
                qb = quarter_band_stats.get(qn, {})
                for bname in bands:
                    st = qb.get(bname)
                    if not st:
                        continue
                    lines.append(f"    {bname}:")
                    lines.append(f"      min: {st['min']}")
                    lines.append(f"      max: {st['max']}")
                    lines.append(f"      mean: {st['mean']:.3f}")
                    lines.append(f"      std: {st['std']:.3f}")
                    lines.append(f"      p5: {st['p5']:.3f}")
                    lines.append(f"      p50: {st['p50']:.3f}")
                    lines.append(f"      p95: {st['p95']:.3f}")
                    lines.append(f"      valid_ratio: {st['valid_ratio']:.4f}")
                    lines.append(f"      count: {st['count']}")
                    lines.append(f"      valid_count: {st['valid_count']}")
            lines.append(f"s1_key: {fn_s1_key}")
            lines.append(f"dates_key: {fn_s1_dates_key}")
            lines.append(f"write_ms: {_elapsed_ms(t_write):.0f}")
            lines.append(f"total_ms: {_elapsed_ms(t_all):.0f}")

            obstore_put_text(store, meta_key, "\n".join(lines) + "\n")
            logger.success(f"S1 metadata sidecar saved: {args.dest.rstrip('/')}/{meta_key}")
        except Exception as e:
            logger.error(f"Failed to write metadata sidecar: {e}")


# --- programmatic entrypoint for Lithops parity ---
def run(
    year: int | str,
    lon: float,
    lat: float,
    X_tile: int | str,
    Y_tile: int | str,
    dest: str,
    expansion: int = 300,
    debug: bool = False,
    run_metadata: bool = False,
    orbit_direction: str = "ASCENDING",
    k_scenes: int = 3,
    coverage_threshold: float = 0.95,
) -> dict:
    import sys
    argv = [
        __file__,
        "--year", str(year),
        "--lon", str(lon),
        "--lat", str(lat),
        "--X_tile", str(X_tile),
        "--Y_tile", str(Y_tile),
        "--dest", dest,
        "--expansion", str(expansion),
        "--orbit-direction", orbit_direction,
        "--k-scenes", str(k_scenes),
        "--coverage-threshold", str(coverage_threshold),
    ]
    if debug:
        argv.append("--debug")
    if run_metadata:
        argv.append("--run-metadata")

    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old

    return {
        "product": "s1_rtc_pc",
        "year": int(year),
        "lon": float(lon),
        "lat": float(lat),
        "X_tile": int(X_tile),
        "Y_tile": int(Y_tile),
        "dest": dest,
        "orbit_direction": orbit_direction,
        "k_scenes": int(k_scenes),
        "coverage_threshold": float(coverage_threshold),
    }


if __name__ == "__main__":
    main()
