#!/usr/bin/env python3
import os
import requests
import numpy as np
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from loguru import logger
import rasterio as rio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from sklearn.ensemble import HistGradientBoostingRegressor

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

import numpy as np
from scipy.ndimage import zoom
from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-6

def _resize2d_nn(arr, H, W):
    """Fast nearest-neighbor resize without extra deps."""
    arr = np.asarray(arr)
    h, w = arr.shape
    yi = (np.linspace(0, h - 1, H)).astype(np.int64)
    xi = (np.linspace(0, w - 1, W)).astype(np.int64)
    return arr[np.ix_(yi, xi)]

def _wrap_angle_pi(a):
    """Wrap radians to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def _make_features(slope, aspect, dem, cos_inc=None):
    """
    Features: slope, sin(aspect), cos(aspect), dem, optional cos_inc
    """
    slope = slope.astype(np.float32)
    aspect = aspect.astype(np.float32)
    dem = dem.astype(np.float32)

    f = [
        slope,
        np.sin(aspect),
        np.cos(aspect),
        dem,
    ]
    if cos_inc is not None:
        f.append(cos_inc.astype(np.float32))
    X = np.stack(f, axis=-1)  # (H,W,F)
    return X

def flatten_s1_tile_hgbm(
    s1,                     # (T,H,W,2) float32 in [0,1] or similar
    slope, aspect, dem,     # (h0,w0) arrays (radians for slope/aspect)
    cos_inc=None,           # optional (h0,w0) or None
    theta_i_deg=32.4,
    max_train=200_000,
    sample_seed=0,
    mask_valid_min=1e-6,
    do_residual=True,
    hgbm_kwargs=None,
):
    """
    Returns:
      s1_flat: (T,H,W,2) float32 flattened as if slope=0
      dbg: dict
    """
    assert s1.ndim == 4 and s1.shape[-1] == 2, f"Expected (T,H,W,2), got {s1.shape}"
    T, H, W, B = s1.shape
    assert B == 2

    # --- resample terrain layers to SAR shape ---
    slope_r  = _resize2d_nn(slope, H, W)
    aspect_r = _resize2d_nn(aspect, H, W)
    dem_r    = _resize2d_nn(dem, H, W)
    cos_r    = _resize2d_nn(cos_inc, H, W) if cos_inc is not None else None

    # Build observed features
    X_obs = _make_features(slope_r, aspect_r, dem_r, cos_r)  # (H,W,F)
    F = X_obs.shape[-1]

    # Build reference-flat features: slope=0, aspect=0 (or keep aspect; it shouldn't matter if slope=0)
    slope0 = np.zeros((H, W), dtype=np.float32)
    aspect0 = np.zeros((H, W), dtype=np.float32)

    if cos_r is not None:
        # on flat terrain, cos_local_inc ~= cos(theta_i)
        cos_ti = np.cos(np.deg2rad(theta_i_deg)).astype(np.float32)
        cos0 = np.full((H, W), cos_ti, dtype=np.float32)
        X_ref = _make_features(slope0, aspect0, dem_r, cos0)
    else:
        X_ref = _make_features(slope0, aspect0, dem_r, None)

    # Flatten spatial dims
    X_obs_2d = X_obs.reshape(-1, F)
    X_ref_2d = X_ref.reshape(-1, F)

    # Valid mask: require SAR signal present (avoid training on nodata/zeros)
    # (you may want to replace this with your real nodata mask if you have it)
    # Use both bands to decide validity
    valid = (s1[..., 0] > mask_valid_min) | (s1[..., 1] > mask_valid_min)  # (T,H,W)
    valid_any = valid.any(axis=0).reshape(-1)  # pixels valid in any timestep

    # Sample training pixels from valid_any
    idx_all = np.flatnonzero(valid_any)
    if idx_all.size == 0:
        raise ValueError("No valid pixels found for training (check mask_valid_min / nodata handling).")

    rng = np.random.default_rng(sample_seed)
    n_train = min(max_train, idx_all.size)
    train_idx = rng.choice(idx_all, size=n_train, replace=False)

    # Prepare outputs
    s1_flat = np.empty_like(s1, dtype=np.float32)

    # Default HGBM params
    if hgbm_kwargs is None:
        hgbm_kwargs = dict(
            max_depth=8,
            learning_rate=0.08,
            max_iter=300,
            min_samples_leaf=30,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=sample_seed,
        )

    dbg = {
        "H": H, "W": W, "F": F,
        "n_train": int(n_train),
        "train_valid_frac": float(n_train / idx_all.size),
        "theta_i_deg": float(theta_i_deg),
    }

    # Train per band (VV, VH) using pooled timesteps (or per-timestep; pooled is usually better)
    # We pool over time by sampling (pixel, time) pairs.
    # Keep it simple: train on median over time to avoid speckle dominating.
    y_med = np.median(s1, axis=0)  # (H,W,2)

    for b in range(2):
        y = y_med[..., b].reshape(-1)

        # training rows
        Xtr = X_obs_2d[train_idx]
        ytr = y[train_idx]

        # Fit modelnnnn
        model = HistGradientBoostingRegressor(**hgbm_kwargs)
        model.fit(Xtr, ytr)

        # Predict observed + reference for ALL pixels
        pred_obs = model.predict(X_obs_2d).astype(np.float32)
        pred_ref = model.predict(X_ref_2d).astype(np.float32)

        # Residual correction: y_flat = y + (pred_ref - pred_obs)
        # Apply this correction to every timestep (preserve temporal variability)
        delta = (pred_ref - pred_obs).reshape(H, W)

        if do_residual:
            s1_flat[..., b] = s1[..., b] + delta[None, ...]
        else:
            # Direct replacement (less stable): every timestep becomes "ref prediction"
            s1_flat[..., b] = pred_ref.reshape(H, W)[None, ...]

        dbg[f"band{b}_delta_stats"] = dict(
            min=float(delta.min()),
            max=float(delta.max()),
            mean=float(delta.mean()),
            p1=float(np.percentile(delta, 1)),
            p99=float(np.percentile(delta, 99)),
        )

    # Clamp to sane range (match your old pipeline style)
    s1_flat = np.clip(s1_flat, 0.0, 1.0)

    return s1_flat, dbg

def to_db(x, eps=EPS):
    x = np.asarray(x, np.float32)
    return 10.0 * np.log10(np.clip(x, eps, None))

def from_db(db):
    return np.power(10.0, db / 10.0).astype(np.float32)

def resize2d(arr, out_h, out_w, order=1):
    """
    Resize a 2D array to (out_h, out_w) using scipy.ndimage.zoom.
    order=1 is bilinear; order=0 nearest; order=3 cubic.
    """
    arr = np.asarray(arr, np.float32)
    in_h, in_w = arr.shape
    zh = out_h / in_h
    zw = out_w / in_w
    return zoom(arr, (zh, zw), order=order)

def maybe_to_radians_slope_aspect(slope, aspect):
    slope = slope.astype(np.float32)
    aspect = aspect.astype(np.float32)

    # If slope looks like degrees (max > ~1.8 rad), convert
    if np.nanmax(slope) > 1.8:
        slope = np.deg2rad(slope)

    # If aspect looks like degrees (max > 2*pi), convert
    if np.nanmax(aspect) > (2*np.pi + 0.5):
        aspect = np.deg2rad(aspect)

    # Wrap aspect into [0, 2pi)
    aspect = np.mod(aspect, 2*np.pi).astype(np.float32)
    return slope, aspect

def build_features(slope_rad, aspect_rad, dem=None,
                   theta_i_deg=32.4, look_dir_rad=np.deg2rad(270.0),
                   cos_local_inc=None):
    """
    Returns X: (H,W,F), and cos_local_inc: (H,W)
    """
    theta_i = np.deg2rad(theta_i_deg).astype(np.float32)

    if cos_local_inc is None:
        cos_local_inc = (
            np.cos(slope_rad) * np.cos(theta_i) +
            np.sin(slope_rad) * np.sin(theta_i) * np.cos(aspect_rad - look_dir_rad)
        ).astype(np.float32)
    else:
        cos_local_inc = cos_local_inc.astype(np.float32)

    asp_rel = (aspect_rad - look_dir_rad).astype(np.float32)

    feats = [
        np.cos(theta_i) * np.ones_like(slope_rad, np.float32),
        np.clip(cos_local_inc, -1.0, 1.0),
        slope_rad.astype(np.float32),
        np.sin(slope_rad).astype(np.float32),
        np.cos(slope_rad).astype(np.float32),
        np.sin(asp_rel).astype(np.float32),
        np.cos(asp_rel).astype(np.float32),
    ]

    if dem is not None:
        dem = dem.astype(np.float32)
        med = np.nanmedian(dem)
        iqr = np.nanpercentile(dem, 75) - np.nanpercentile(dem, 25)
        dem_scaled = (dem - med) / (iqr + 1e-6)
        feats.append(dem_scaled.astype(np.float32))

    X = np.stack(feats, axis=-1)  # (H,W,F)
    return X, cos_local_inc

def sample_points(X, y, mask, max_points=200_000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        raise RuntimeError("No valid pixels for training after masking.")
    take = min(max_points, idx.size)
    sel = rng.choice(idx, size=take, replace=False)
    Xs = X.reshape(-1, X.shape[-1])[sel]
    ys = y.ravel()[sel]
    return Xs, ys

def fit_hgbm(Xs, ys, random_state=0):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    ).fit(Xs, ys)

def terrain_correct_s1_hickle(
    s1, slope, aspect, dem=None,
    theta_i_deg=32.4,
    orbit_state="DESCENDING",         # or "ASCENDING"
    cos_local_inc=None,               # optional 2D array on slope grid, or None
    max_train_points=200_000,
    rng_seed=0,
    use_all_timesteps_for_training=True,
    per_timestep_training=False,      # set True only if you want separate models per t
    correction_db_cap=5.0,
):
    """
    s1: (T, H, W, 2) float in [0,1] or linear gamma0-ish
    slope/aspect/dem: (h0, w0) arrays (e.g. 200x200)
    Returns:
      s1_corr: (T,H,W,2) float32 corrected (linear)
      debug: dict with bias maps, refs, etc.
    """

    s1 = np.asarray(s1, np.float32)
    assert s1.ndim == 4 and s1.shape[-1] == 2, "Expected s1 shape (T,H,W,2)"
    T, H, W, _ = s1.shape

    # Resize slope/aspect/dem to match S1 spatial grid
    print(slope.shape, aspect.shape, type(slope), type(aspect))
    slope_r = resize2d(slope, H, W, order=1)
    aspect_r = resize2d(aspect, H, W, order=1)
    slope_r, aspect_r = maybe_to_radians_slope_aspect(slope_r, aspect_r)

    dem_r = None
    if dem is not None:
        dem_r = resize2d(dem, H, W, order=1)

    cos_r = None
    if cos_local_inc is not None:
        # allow passing a 200x200 cos_local; resize it
        cos_r = resize2d(cos_local_inc, H, W, order=1).astype(np.float32)

    # look direction from orbit
    if str(orbit_state).upper().startswith("ASC"):
        look_dir_rad = np.deg2rad(90.0).astype(np.float32)   # east
    else:
        look_dir_rad = np.deg2rad(270.0).astype(np.float32)  # west

    # Build features once (same for all timesteps)
    X, cos_local = build_features(
        slope_rad=slope_r,
        aspect_rad=aspect_r,
        dem=dem_r,
        theta_i_deg=theta_i_deg,
        look_dir_rad=look_dir_rad,
        cos_local_inc=cos_r,
    )

    # Masks
    valid_geom = np.isfinite(slope_r) & np.isfinite(aspect_r) & np.isfinite(cos_local)
    if dem_r is not None:
        valid_geom &= np.isfinite(dem_r)

    valid_cos = (cos_local > 0.15) & (cos_local < 0.9999)

    slope_deg = np.rad2deg(slope_r)
    flat = valid_geom & valid_cos & (slope_deg < 3.0)
    if np.count_nonzero(flat) < 10_000:
        flat = valid_geom & valid_cos & (slope_deg < 5.0)

    # Output arrays
    s1_corr = np.zeros_like(s1, dtype=np.float32)

    # Debug holders (optionally store one bias map shared across time)
    debug = {
        "theta_i_deg": float(theta_i_deg),
        "orbit_state": str(orbit_state),
        "cos_local_stats": {
            "min": float(np.nanmin(cos_local)),
            "mean": float(np.nanmean(cos_local)),
            "max": float(np.nanmax(cos_local)),
            "p5": float(np.nanpercentile(cos_local, 5)),
            "p50": float(np.nanpercentile(cos_local, 50)),
            "p95": float(np.nanpercentile(cos_local, 95)),
        },
        "flat_ratio": float(np.count_nonzero(flat) / flat.size),
        "models": {},
        "bias_db": None,   # (H,W,2) if shared
    }

    # Helper to train/predict for one set of vv/vh images
    def train_and_apply(vv_img, vh_img, seed_offset=0):
        vv_db = to_db(vv_img)
        vh_db = to_db(vh_img)

        valid_vv = np.isfinite(vv_db) & (vv_img > 0)
        valid_vh = np.isfinite(vh_db) & (vh_img > 0)

        vv_ref = np.nanmedian(vv_db[flat & valid_vv]) if np.any(flat & valid_vv) else np.nanmedian(vv_db[valid_vv])
        vh_ref = np.nanmedian(vh_db[flat & valid_vh]) if np.any(flat & valid_vh) else np.nanmedian(vh_db[valid_vh])

        vv_target = vv_db - vv_ref
        vh_target = vh_db - vh_ref

        train_mask_vv = valid_geom & valid_cos & valid_vv
        train_mask_vh = valid_geom & valid_cos & valid_vh

        Xv, yv = sample_points(X, vv_target, train_mask_vv, max_points=max_train_points, rng_seed=rng_seed + seed_offset)
        Xh, yh = sample_points(X, vh_target, train_mask_vh, max_points=max_train_points, rng_seed=rng_seed + seed_offset + 1)

        mvv = fit_hgbm(Xv, yv, random_state=rng_seed + seed_offset)
        mvh = fit_hgbm(Xh, yh, random_state=rng_seed + seed_offset + 1)

        Xflat = X.reshape(-1, X.shape[-1])
        vv_bias = mvv.predict(Xflat).reshape(H, W).astype(np.float32)
        vh_bias = mvh.predict(Xflat).reshape(H, W).astype(np.float32)

        vv_bias = np.clip(vv_bias, -correction_db_cap, correction_db_cap)
        vh_bias = np.clip(vh_bias, -correction_db_cap, correction_db_cap)

        vv_corr_db = vv_db - vv_bias
        vh_corr_db = vh_db - vh_bias

        vv_corr = from_db(vv_corr_db)
        vh_corr = from_db(vh_corr_db)

        # keep zeros as zeros
        vv_corr = np.where(valid_vv, vv_corr, 0.0).astype(np.float32)
        vh_corr = np.where(valid_vh, vh_corr, 0.0).astype(np.float32)

        return vv_corr, vh_corr, vv_bias, vh_bias, float(vv_ref), float(vh_ref), mvv, mvh

    # Strategy:
    # 1) Train once using all timesteps stacked (preferred)
    # 2) Or train on t=0 and apply to all
    # 3) Or per-timestep training (expensive)
    if per_timestep_training:
        # train separately per timestep
        bias_maps = []
        refs = []
        for t in range(T):
            vv = s1[t, :, :, 0]
            vh = s1[t, :, :, 1]
            vv_c, vh_c, vv_b, vh_b, vv_ref, vh_ref, mvv, mvh = train_and_apply(vv, vh, seed_offset=10*t)
            s1_corr[t, :, :, 0] = vv_c
            s1_corr[t, :, :, 1] = vh_c
            bias_maps.append(np.stack([vv_b, vh_b], axis=-1))
            refs.append((vv_ref, vh_ref))
        debug["bias_db"] = np.stack(bias_maps, axis=0)  # (T,H,W,2)
        debug["refs_db"] = refs
        return s1_corr, debug

    if use_all_timesteps_for_training:
        # Make a pooled training image by sampling pixels from all timesteps,
        # but simplest is: choose a representative composite (median in dB)
        vv_stack_db = to_db(s1[..., 0])
        vh_stack_db = to_db(s1[..., 1])

        # Median across time for stability
        vv_med_db = np.nanmedian(np.where(vv_stack_db > -1e9, vv_stack_db, np.nan), axis=0).astype(np.float32)
        vh_med_db = np.nanmedian(np.where(vh_stack_db > -1e9, vh_stack_db, np.nan), axis=0).astype(np.float32)

        vv_med = from_db(vv_med_db)
        vh_med = from_db(vh_med_db)

        vv_c, vh_c, vv_b, vh_b, vv_ref, vh_ref, mvv, mvh = train_and_apply(vv_med, vh_med, seed_offset=0)

        # Apply the same bias maps to every timestep in dB
        vv_bias = vv_b
        vh_bias = vh_b
        debug["bias_db"] = np.stack([vv_bias, vh_bias], axis=-1)
        debug["refs_db"] = (vv_ref, vh_ref)
        debug["models"]["vv"] = mvv
        debug["models"]["vh"] = mvh

        for t in range(T):
            vv_db = to_db(s1[t, :, :, 0])
            vh_db = to_db(s1[t, :, :, 1])

            vv_corr = from_db(vv_db - vv_bias)
            vh_corr = from_db(vh_db - vh_bias)

            vv_valid = (s1[t, :, :, 0] > 0) & np.isfinite(vv_db)
            vh_valid = (s1[t, :, :, 1] > 0) & np.isfinite(vh_db)

            s1_corr[t, :, :, 0] = np.where(vv_valid, vv_corr, 0.0).astype(np.float32)
            s1_corr[t, :, :, 1] = np.where(vh_valid, vh_corr, 0.0).astype(np.float32)

        return s1_corr, debug

    # Otherwise: train on first timestep and apply to all
    vv0 = s1[0, :, :, 0]
    vh0 = s1[0, :, :, 1]
    vv_c, vh_c, vv_b, vh_b, vv_ref, vh_ref, mvv, mvh = train_and_apply(vv0, vh0, seed_offset=0)
    debug["bias_db"] = np.stack([vv_b, vh_b], axis=-1)
    debug["refs_db"] = (vv_ref, vh_ref)
    debug["models"]["vv"] = mvv
    debug["models"]["vh"] = mvh

    for t in range(T):
        vv_db = to_db(s1[t, :, :, 0])
        vh_db = to_db(s1[t, :, :, 1])
        s1_corr[t, :, :, 0] = np.where(s1[t, :, :, 0] > 0, from_db(vv_db - vv_b), 0.0)
        s1_corr[t, :, :, 1] = np.where(s1[t, :, :, 1] > 0, from_db(vh_db - vh_b), 0.0)

    return s1_corr.astype(np.float32), debug


def calculate_slope_aspect(dem: np.ndarray, transform, dem_crs) -> tuple:
    """Calculate slope and aspect from DEM.

    Properly handles geographic (lat/lon) vs projected CRS by converting
    pixel spacing to meters before computing gradients.
    """
    logger.debug("Calculating slope and aspect from DEM")

    # Extract pixel sizes from transform
    pixel_x = abs(transform.a)  # pixel width (may be degrees or meters)
    pixel_y = abs(transform.e)  # pixel height (may be degrees or meters)

    # Check if CRS is geographic (degrees) vs projected (meters)
    crs_obj = CRS.from_user_input(dem_crs) if dem_crs else None
    is_geographic = crs_obj.is_geographic if crs_obj else False

    if is_geographic:
        # Convert degrees to meters using latitude-aware calculation
        # Get center latitude of the DEM for conversion
        # transform.f is top-left Y, transform.e is negative pixel height
        center_lat = transform.f + (transform.e * dem.shape[0] / 2)

        # Meters per degree (approximate, varies with latitude)
        meters_per_degree_lat = 111320.0  # approximately constant
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))

        # Convert pixel spacing from degrees to meters
        resolution_y = pixel_y * meters_per_degree_lat
        resolution_x = pixel_x * meters_per_degree_lon

        logger.debug(f"Geographic CRS detected at lat={center_lat:.2f}°")
        logger.debug(f"Pixel spacing: {pixel_y:.6f}° x {pixel_x:.6f}° = {resolution_y:.1f}m x {resolution_x:.1f}m")
    else:
        # Projected CRS - pixel spacing already in meters (or linear units)
        resolution_y = pixel_y
        resolution_x = pixel_x
        logger.debug(f"Projected CRS - pixel spacing: {resolution_y:.1f}m x {resolution_x:.1f}m")

    # Calculate gradients with correct spacing (dy uses Y spacing, dx uses X spacing)
    dy, dx = np.gradient(dem, resolution_y, resolution_x)

    # Calculate slope (in radians)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))

    # Calculate aspect (in radians, 0 = North, clockwise)
    aspect = np.arctan2(-dx, dy)
    aspect = np.where(aspect < 0, aspect + 2*np.pi, aspect)

    logger.debug(f"Slope range: {np.degrees(slope.min()):.1f} to {np.degrees(slope.max()):.1f} degrees")

    return slope, aspect

def download_dem_for_bbox(
    bbox,
    out_dem_tif: str,
    bbox_crs="EPSG:4326",             # CRS of bbox (default lon/lat)
    demtype: str = "COP30",           # e.g., "COP30", "SRTMGL1", etc.
    api_key: str | None = None,       # OpenTopography API key
    match_grid: bool = False,         # if True: warp to target grid below
    target_crs=None,                  # required if match_grid=True
    target_transform=None,            # required if match_grid=True (Affine)
    target_width=None,                # required if match_grid=True
    target_height=None,               # required if match_grid=True
    resampling: Resampling = Resampling.bilinear,
    timeout: int = 300,
):
    """
    Download a DEM covering `bbox` from OpenTopography GlobalDEM API.

    Parameters
    ----------
    bbox : tuple
        (left, bottom, right, top) in bbox_crs coordinates.
        OpenTopography requires a WGS84 bbox (lon/lat degrees); conversion happens automatically.
    out_dem_tif : str
        Output GeoTIFF path.
    bbox_crs : str or rasterio CRS
        CRS of bbox; will be transformed to EPSG:4326 for the API call.
    match_grid : bool
        If False: save downloaded GeoTIFF as-is.
        If True: reproject/resample to an explicit target grid (target_* args).
    target_* : defines output grid if match_grid=True
        - target_crs: output CRS
        - target_transform: affine transform
        - target_width/target_height: raster dimensions
    """

    if api_key is None:
        api_key = os.environ.get("OPENTOPO_API_KEY") or os.environ.get("OPENTOPO_APIKEY")
    if not api_key:
        raise ValueError(
            "OpenTopography API key missing. Pass api_key=... or set env OPENTOPO_API_KEY."
        )

    left, bottom, right, top = bbox

    # Convert bbox to EPSG:4326 lon/lat for OpenTopography
    west, south, east, north = transform_bounds(
        bbox_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
    )

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

    os.makedirs(os.path.dirname(out_dem_tif) or ".", exist_ok=True)

    # --- Simple mode: save as-is ---
    if not match_grid:
        with open(out_dem_tif, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return out_dem_tif

    # --- Match-grid mode: warp to explicit target grid ---
    if target_crs is None or target_transform is None or target_width is None or target_height is None:
        raise ValueError(
            "match_grid=True requires target_crs, target_transform, target_width, target_height."
        )

    dem_bytes = r.content
    with MemoryFile(dem_bytes) as mem:
        with mem.open() as dem:
            dst = np.zeros((int(target_height), int(target_width)), dtype=np.float32)

            reproject(
                source=rasterio.band(dem, 1),
                destination=dst,
                src_transform=dem.transform,
                src_crs=dem.crs,
                src_nodata=dem.nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

            out_profile = {
                "driver": "GTiff",
                "height": int(target_height),
                "width": int(target_width),
                "count": 1,
                "dtype": "float32",
                "crs": target_crs,
                "transform": target_transform,
                "nodata": np.nan,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }

            with rasterio.open(out_dem_tif, "w", **out_profile) as out:
                out.write(dst, 1)

    return out_dem_tif

def write_from_bbox(path, arr2d, bbox, crs="EPSG:4326", nodata=None, compress="deflate"):
    H, W = arr2d.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)
    profile = dict(
        driver="GTiff",
        height=H, width=W, count=1,
        dtype=str(arr2d.dtype),
        crs=crs,
        transform=transform,
        compress=compress,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr2d, 1)


def make_target_grid_from_bbox(
    bbox,
    bbox_crs="EPSG:4326",
    target_crs="EPSG:4326",
    resolution=None,   # (xres, yres) in target_crs units; e.g. (0.0002777778, 0.0002777778) for ~30m at equator
    width=None,
    height=None,
):
    """
    Convenience helper to define a target grid from a bbox.

    Provide either:
      - resolution=(xres, yres)  -> computes width/height
      - OR width and height      -> computes transform

    Returns: (target_crs, transform, width, height)
    """
    left, bottom, right, top = bbox
    t_left, t_bottom, t_right, t_top = transform_bounds(
        bbox_crs, target_crs, left, bottom, right, top, densify_pts=21
    )

    if resolution is not None:
        xres, yres = resolution
        width = int(np.ceil((t_right - t_left) / xres))
        height = int(np.ceil((t_top - t_bottom) / abs(yres)))
        transform = from_bounds(t_left, t_bottom, t_right, t_top, width, height)
        return target_crs, transform, width, height

    if width is None or height is None:
        raise ValueError("Provide either resolution=(xres,yres) or both width and height.")

    transform = from_bounds(t_left, t_bottom, t_right, t_top, int(width), int(height))
    return target_crs, transform, int(width), int(height)


if __name__ == "__main__":
    import hickle as hkl
    # Example:
    # export OPENTOPO_API_KEY="3d88154f91b04bf896495d9a079039c7"
    x = '2349'
    y = '1148' 
    year = '2024'
    bbox = [36.83332222222222, 10.222222222222221, 36.88887777777778, 10.277777777777779]

    s1_path = f'../project-monitoring/tiles/{year}/{x}/{y}/raw/s1/{x}X{y}Y.hkl'
    rgb_path = f'../project-monitoring/tiles/{year}/{x}/{y}/raw/s2_10/{x}X{y}Y.hkl'
    output_path = f"../project-monitoring/tiles/{year}/{x}/{y}/raw/dem.tif"
    if not os.path.exists(output_path):
    
        download_dem_for_bbox(bbox, f"../project-monitoring/tiles/{year}/{x}/{y}/raw/dem.tif", demtype="COP30", match_grid=False)
        print(f"Wrote: {output_path}")

    dem = rio.open(output_path)
    slope, aspect = calculate_slope_aspect(dem.read(1), dem.transform, dem.crs)
    s1 = hkl.load(s1_path)
    rgb = hkl.load(rgb_path) / 65535
    s1 = s1 / 65535
    
    '''
    s1_corr, dbg = terrain_correct_s1_hickle(
        s1=s1,
        slope=slope,
        aspect=aspect,
        dem=dem.read(1),                     # optional
        theta_i_deg=32.4,
        orbit_state="DESCENDING",     # or "ASCENDING"
        max_train_points=50_000,
        rng_seed=0,
        use_all_timesteps_for_training=True,  # recommended
        per_timestep_training=False,
        correction_db_cap=10.0
    )

    print("corr shape:", s1_corr.shape)
    print("cos_local stats:", dbg["cos_local_stats"])
    print("flat_ratio:", dbg["flat_ratio"])
    '''

    s1_flat, dbg = flatten_s1_tile_hgbm(
        s1=s1,
        slope=slope,          # radians
        aspect=aspect,        # radians
        dem=dem.read(1),              # meters
        cos_inc=None,      # optional; pass None if you don't have it
        theta_i_deg=32.4,
        max_train=50_000,
    )


    # Example: save corrected VV at timestep 0
    #vv0 = s1_corr[0, :, :, 0].astype(np.float32)
    write_from_bbox("vv0_orr.tif", np.median(s1[..., 0], axis = 0), bbox=bbox, crs="EPSG:4326", nodata=65536.0)
    write_from_bbox("vv0_corr.tif", np.median(s1_flat[..., 0], axis = 0), bbox=bbox, crs="EPSG:4326", nodata=65536.0)
    write_from_bbox("rgb_orr.tif", np.median(rgb[..., 0], axis = 0), bbox=bbox, crs="EPSG:4326", nodata=65536.0)

    # Save bias map (dB) for VV
    #vv_bias_db = dbg["bias_db"][:, :, 0].astype(np.float32)
    #write_from_bbox("vv_bias_db.tif", vv_bias_db, bbox=bbx, crs="EPSG:4326", nodata=np.nan)

