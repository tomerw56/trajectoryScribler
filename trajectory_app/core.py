from __future__ import annotations

from .logging_config import get_logger
import math
from typing import Iterable
import numpy as np

from .dubins import DubinsFitResult, fit_freehand_dubins
from .models import (
    AltitudeMode,
    CovarianceSamples,
    DerivedVelocityModel,
    MotionConfig,
    MotionSamples,
    TerrainConfig,
    TrajectoryConfig,
    TrajectoryResult,
)


logger = get_logger(__name__)

def _gaussian_feature(xg, yg, cx, cy, amplitude, sx, sy):
    xt = ((xg - cx) ** 2) / (2.0 * sx**2)
    yt = ((yg - cy) ** 2) / (2.0 * sy**2)
    return amplitude * np.exp(-(xt + yt))


def generate_terrain(config: TerrainConfig) -> np.ndarray:
    """Port of the supplied height-map generator, without project-specific model dependencies."""
    if config.grid_size < 20 or config.grid_size > 300:
        raise ValueError("grid_size must be between 20 and 300")
    if config.x_max <= config.x_min or config.y_max <= config.y_min:
        raise ValueError("Terrain max bounds must be greater than min bounds")
    if config.max_sigma < config.min_sigma or config.min_sigma <= 0:
        raise ValueError("Sigma range is invalid")

    rng = np.random.default_rng(config.seed)
    xv = np.linspace(config.x_min, config.x_max, config.grid_size)
    yv = np.linspace(config.y_min, config.y_max, config.grid_size)
    xg, yg = np.meshgrid(xv, yv)
    h = np.full_like(xg, config.base_height, dtype=float)

    for count, amin, amax, sign in (
        (config.mountain_count, config.mountain_height_min, config.mountain_height_max, 1.0),
        (config.valley_count, config.valley_depth_min, config.valley_depth_max, -1.0),
    ):
        for _ in range(count):
            cx = float(rng.uniform(config.x_min, config.x_max))
            cy = float(rng.uniform(config.y_min, config.y_max))
            amp = float(sign * rng.uniform(amin, amax))
            sx = float(rng.uniform(config.min_sigma, config.max_sigma))
            sy = float(rng.uniform(config.min_sigma, config.max_sigma))
            h += _gaussian_feature(xg, yg, cx, cy, amp, sx, sy)

    if config.noise_amplitude > 0:
        h += rng.normal(0.0, config.noise_amplitude, size=h.shape)
    return h.astype(np.float64)


def bilinear_height(terrain, x_m, y_m, cfg: TerrainConfig):
    rows, cols = terrain.shape
    x = (np.asarray(x_m, dtype=float) - cfg.x_min) / cfg.x_cell_size
    y = (np.asarray(y_m, dtype=float) - cfg.y_min) / cfg.y_cell_size
    x = np.clip(x, 0.0, cols - 1.0); y = np.clip(y, 0.0, rows - 1.0)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, cols - 1); y1 = np.minimum(y0 + 1, rows - 1)
    tx = x - x0; ty = y - y0
    z00 = terrain[y0, x0]; z10 = terrain[y0, x1]
    z01 = terrain[y1, x0]; z11 = terrain[y1, x1]
    return (z00 * (1-tx) + z10 * tx) * (1-ty) + (z01 * (1-tx) + z11 * tx) * ty


def terrain_surface_position(
    terrain: np.ndarray,
    terrain_cfg: TerrainConfig,
    x_m: float,
    y_m: float,
) -> np.ndarray:
    """Return a clamped XYZ point lying exactly on the terrain surface."""
    x = float(np.clip(x_m, terrain_cfg.x_min, terrain_cfg.x_max))
    y = float(np.clip(y_m, terrain_cfg.y_min, terrain_cfg.y_max))
    z = float(bilinear_height(terrain, x, y, terrain_cfg))
    return np.array([x, y, z], dtype=float)


def view_sector_span_deg(start_deg: float, end_deg: float) -> float:
    """Counter-clockwise sector span, supporting wraparound such as 330 -> 30."""
    raw = float(end_deg) - float(start_deg)
    if abs(raw) < 1e-12:
        return 0.0
    return raw % 360.0


def view_sector_boundary_xy(
    origin_x: float,
    origin_y: float,
    radius_m: float,
    start_deg: float,
    end_deg: float,
    max_step_deg: float = 4.0,
) -> np.ndarray:
    """Return arc boundary XY points for a cameraman view sector."""
    if radius_m <= 0:
        return np.empty((0, 2), dtype=float)
    span = view_sector_span_deg(start_deg, end_deg)
    if span <= 0:
        return np.empty((0, 2), dtype=float)
    count = max(2, int(math.ceil(span / max(0.1, max_step_deg))) + 1)
    angles = np.deg2rad(np.linspace(start_deg, start_deg + span, count))
    return np.column_stack(
        (
            origin_x + radius_m * np.cos(angles),
            origin_y + radius_m * np.sin(angles),
        )
    )


def resample_polyline_2d(points: Iterable[Iterable[float]], spacing_m: float) -> np.ndarray:
    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("At least two XY scribble points are required.")
    if spacing_m <= 0: raise ValueError("path resolution must be > 0")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-12]
    pts = pts[keep]
    if len(pts) < 2: raise ValueError("Scribble is too short")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]; total = float(cum[-1])
    d = np.arange(0.0, total, spacing_m)
    if len(d) == 0 or not np.isclose(d[-1], total): d = np.r_[d, total]
    return np.c_[np.interp(d, cum, pts[:,0]), np.interp(d, cum, pts[:,1])]


def smooth_polyline_2d(points: np.ndarray, passes: int) -> np.ndarray:
    out = points.copy()
    for _ in range(max(0, passes)):
        if len(out) < 3: break
        old = out.copy(); out[1:-1] = .25*old[:-2] + .5*old[1:-1] + .25*old[2:]
    return out


def _cruise_profile(required, xy, angle_deg):
    slope = math.tan(math.radians(float(np.clip(angle_deg, .1, 89.0))))
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    left = required.astype(float).copy()
    for i in range(1, len(left)): left[i] = max(left[i], left[i-1] - slope*ds[i-1])
    right = required.astype(float).copy()
    for i in range(len(right)-2, -1, -1): right[i] = max(right[i], right[i+1] - slope*ds[i])
    return np.maximum(left, right)


def _build_trajectory_from_xy(
    xy: np.ndarray,
    terrain,
    terrain_cfg,
    cfg: TrajectoryConfig,
    *,
    reject_out_of_bounds: bool = False,
) -> TrajectoryResult:
    xy = np.asarray(xy, dtype=float).copy()
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError("Trajectory requires at least two XY points")

    outside = (
        (xy[:, 0] < terrain_cfg.x_min)
        | (xy[:, 0] > terrain_cfg.x_max)
        | (xy[:, 1] < terrain_cfg.y_min)
        | (xy[:, 1] > terrain_cfg.y_max)
    )
    if reject_out_of_bounds and bool(np.any(outside)):
        raise ValueError(
            "The fitted Dubins path leaves the terrain bounds. "
            "Reduce the minimum turn radius or redraw farther from the edge."
        )

    if not reject_out_of_bounds:
        xy[:, 0] = np.clip(xy[:, 0], terrain_cfg.x_min, terrain_cfg.x_max)
        xy[:, 1] = np.clip(xy[:, 1], terrain_cfg.y_min, terrain_cfg.y_max)

    tz = bilinear_height(terrain, xy[:, 0], xy[:, 1], terrain_cfg)
    if cfg.altitude_mode == AltitudeMode.TERRAIN_CLEARANCE:
        z = tz + cfg.clearance_m
    else:
        required = np.maximum(cfg.cruise_altitude_m, tz + cfg.clearance_m)
        z = _cruise_profile(required, xy, cfg.max_climb_angle_deg)

    xyz = np.c_[xy, z]
    cumulative = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)),
    ]
    return TrajectoryResult(xy, xyz, tz, cumulative)


def build_trajectory(raw, terrain, terrain_cfg, cfg: TrajectoryConfig) -> TrajectoryResult:
    """Build the original smoothed-freehand trajectory."""

    xy = resample_polyline_2d(raw, cfg.path_resolution_m)
    xy = smooth_polyline_2d(xy, cfg.smoothing_passes)
    xy = resample_polyline_2d(xy, cfg.path_resolution_m)
    return _build_trajectory_from_xy(
        xy,
        terrain,
        terrain_cfg,
        cfg,
        reject_out_of_bounds=False,
    )


def build_dubins_trajectory(
    raw,
    terrain,
    terrain_cfg,
    cfg: TrajectoryConfig,
) -> tuple[TrajectoryResult, DubinsFitResult]:
    """Fit a piecewise Dubins XY path, then apply the existing altitude model."""

    fit = fit_freehand_dubins(
        raw,
        min_turn_radius_m=cfg.dubins_min_turn_radius_m,
        path_resolution_m=cfg.path_resolution_m,
        smoothing_passes=cfg.smoothing_passes,
    )
    trajectory = _build_trajectory_from_xy(
        fit.xy,
        terrain,
        terrain_cfg,
        cfg,
        reject_out_of_bounds=True,
    )
    return trajectory, fit



def _wrapped_angle_delta(angle: np.ndarray) -> np.ndarray:
    """Return wrapped successive angle differences in [-pi, pi]."""
    return (np.diff(angle) + np.pi) % (2.0 * np.pi) - np.pi


def derive_velocity_model(
    trajectory: TrajectoryResult,
    speed_mps: float,
    path_resolution_m: float,
) -> DerivedVelocityModel:
    """Derive a read-only motion summary from the final generated trajectory.

    The path itself is not modified here. XY curvature is measured on a
    coarser, additionally smoothed analysis copy so display-resolution
    quantization and tiny local wiggles do not dominate the extrema.
    """
    speed = max(0.0, float(speed_mps))
    xy = np.asarray(trajectory.xy, dtype=float)
    xyz = np.asarray(trajectory.xyz, dtype=float)

    if len(xy) < 2 or len(xyz) < 2:
        return DerivedVelocityModel(
            speed_mps=speed,
            min_turn_radius_m=math.inf,
            max_turn_rate_deg_s=0.0,
            max_lateral_accel_mps2=0.0,
            max_climb_angle_deg=0.0,
            max_descent_angle_deg=0.0,
            max_climb_rate_mps=0.0,
            max_descent_rate_mps=0.0,
        )

    # Analyze XY geometry at a deliberately coarser resolution than rendering.
    # This is analysis-only and does not alter the displayed/generated path.
    analysis_spacing = max(0.50, float(path_resolution_m) * 4.0)
    analysis_xy = resample_polyline_2d(xy, analysis_spacing)
    if len(analysis_xy) >= 3:
        analysis_xy = smooth_polyline_2d(analysis_xy, passes=2)
        analysis_xy = resample_polyline_2d(analysis_xy, analysis_spacing)

    max_curvature = 0.0
    if len(analysis_xy) >= 3:
        delta = np.diff(analysis_xy, axis=0)
        ds = np.linalg.norm(delta, axis=1)
        good = ds > 1e-9

        if np.count_nonzero(good) >= 2:
            headings = np.arctan2(delta[:, 1], delta[:, 0])
            dtheta = np.abs(_wrapped_angle_delta(headings))
            local_ds = 0.5 * (ds[:-1] + ds[1:])
            valid = local_ds > 1e-9
            if np.any(valid):
                curvature = np.zeros_like(local_ds)
                curvature[valid] = dtheta[valid] / local_ds[valid]
                finite = curvature[np.isfinite(curvature)]
                if len(finite):
                    max_curvature = float(np.max(finite))

    if max_curvature <= 1e-12:
        min_turn_radius = math.inf
        max_turn_rate_deg_s = 0.0
        max_lateral_accel = 0.0
    else:
        min_turn_radius = 1.0 / max_curvature
        max_turn_rate_deg_s = math.degrees(speed * max_curvature)
        max_lateral_accel = speed * speed * max_curvature

    # Vertical characteristics come from the actual final 3D trajectory.
    dxyz = np.diff(xyz, axis=0)
    ds_xy = np.linalg.norm(dxyz[:, :2], axis=1)
    dz = dxyz[:, 2]
    segment_length = np.linalg.norm(dxyz, axis=1)
    valid_segment = segment_length > 1e-9

    climb_angles = np.zeros(len(dxyz), dtype=float)
    climb_angles[valid_segment] = np.arctan2(
        dz[valid_segment],
        ds_xy[valid_segment],
    )

    positive = climb_angles[climb_angles > 0.0]
    negative = climb_angles[climb_angles < 0.0]

    max_climb_rad = float(np.max(positive)) if len(positive) else 0.0
    max_descent_rad = float(abs(np.min(negative))) if len(negative) else 0.0

    return DerivedVelocityModel(
        speed_mps=speed,
        min_turn_radius_m=float(min_turn_radius),
        max_turn_rate_deg_s=float(max_turn_rate_deg_s),
        max_lateral_accel_mps2=float(max_lateral_accel),
        max_climb_angle_deg=float(math.degrees(max_climb_rad)),
        max_descent_angle_deg=float(math.degrees(max_descent_rad)),
        max_climb_rate_mps=float(speed * math.sin(max_climb_rad)),
        max_descent_rate_mps=float(speed * math.sin(max_descent_rad)),
    )


def fixed_velocity_covariance_from_model(
    model: DerivedVelocityModel,
    *,
    reference_dt_sec: float = 1.0,
    sigma_factor: float = 3.0,
) -> np.ndarray:
    """Build one fixed 3x3 velocity covariance from a derived motion model.

    Interpretation:
    - the derived motion envelope is treated as ``sigma_factor`` sigma,
    - horizontal uncertainty is isotropic so the matrix is all-purpose and
      does not depend on the target's current heading,
    - the turn capability is converted to a lateral velocity excursion over a
      fixed reference interval,
    - vertical sigma uses the larger observed climb/descent rate.

    This is a modeling convention, not a statistically unique consequence of
    the trajectory.
    """
    dt = float(reference_dt_sec)
    sigma_n = float(sigma_factor)
    if dt <= 0.0:
        raise ValueError("reference_dt_sec must be > 0")
    if sigma_n <= 0.0:
        raise ValueError("sigma_factor must be > 0")

    speed = max(0.0, float(model.speed_mps))
    omega = math.radians(max(0.0, float(model.max_turn_rate_deg_s)))

    # Limit the one-second heading excursion used by this simple model to 90°.
    # Beyond that point sin(theta) would decrease again and stop representing
    # "more maneuverable means more uncertain".
    heading_excursion = min(omega * dt, math.pi / 2.0)
    sigma_xy = speed * math.sin(heading_excursion) / sigma_n

    vertical_envelope = max(
        0.0,
        float(model.max_climb_rate_mps),
        float(model.max_descent_rate_mps),
    )
    sigma_z = vertical_envelope / sigma_n

    return np.diag(
        np.square(
            np.array(
                [sigma_xy, sigma_xy, sigma_z],
                dtype=float,
            )
        )
    )



def sample_motion(traj, terrain, terrain_cfg, motion: MotionConfig) -> MotionSamples:
    if motion.speed_mps <= 0 or motion.sample_dt_sec <= 0:
        raise ValueError("speed and dt must be > 0")
    cum = traj.cumulative_distance_3d; total = float(cum[-1])
    if total <= 1e-12: raise ValueError("Trajectory is too short")
    step = motion.speed_mps * motion.sample_dt_sec
    d = np.arange(0.0, total, step)
    if len(d)==0 or not np.isclose(d[-1], total): d=np.r_[d,total]
    pos = np.column_stack([np.interp(d,cum,traj.xyz[:,a]) for a in range(3)])
    idx=np.clip(np.searchsorted(cum,d,side="right")-1,0,len(traj.xyz)-2)
    dv=traj.xyz[idx+1]-traj.xyz[idx]; dl=np.linalg.norm(dv,axis=1)
    direction=np.zeros_like(dv); good=dl>1e-12; direction[good]=dv[good]/dl[good,None]
    vel=direction*motion.speed_mps
    t=d/motion.speed_mps
    tz=bilinear_height(terrain,pos[:,0],pos[:,1],terrain_cfg)
    return MotionSamples(t,d,pos,vel,tz)


def sample_velocity_covariance(samples: MotionSamples, motion: MotionConfig) -> CovarianceSamples:
    """Rolling covariance of recent [Vx,Vy,Vz], emitted at an independent covariance sample rate."""
    if motion.covariance_sample_rate_sec <= 0 or motion.covariance_window_sec <= 0:
        raise ValueError("Covariance sample rate and window must be > 0")
    emit_times=np.arange(0.0,float(samples.time_sec[-1])+1e-9,motion.covariance_sample_rate_sec)
    mats=[]
    for t in emit_times:
        lo=t-motion.covariance_window_sec
        mask=(samples.time_sec>=lo)&(samples.time_sec<=t+1e-12)
        vals=samples.velocity[mask]
        if len(vals)<2:
            mats.append(np.zeros((3,3),dtype=float))
        else:
            mats.append(np.cov(vals,rowvar=False,ddof=1))
    return CovarianceSamples(emit_times,np.asarray(mats,dtype=float))


def predict_position_ellipse_xy(
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    velocity_covariance: np.ndarray,
    prediction_dt_sec: float,
    *,
    sigma_scale: float = 1.0,
    point_count: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal future XY center and a covariance ellipse.

    Assumptions for this simple visualization:
    - current position is exact,
    - velocity is constant over ``prediction_dt_sec``,
    - velocity covariance is constant over the same horizon.

    Therefore:
        center_xy = p_xy + v_xy * dt
        position_cov_xy = velocity_cov_xy * dt**2

    The returned ellipse is a 1-sigma ellipse when ``sigma_scale=1``.
    """
    if prediction_dt_sec < 0:
        raise ValueError("prediction_dt_sec must be >= 0")
    if sigma_scale < 0:
        raise ValueError("sigma_scale must be >= 0")
    if point_count < 8:
        raise ValueError("point_count must be >= 8")

    position = np.asarray(current_position, dtype=float).reshape(3)
    velocity = np.asarray(current_velocity, dtype=float).reshape(3)
    covariance = np.asarray(velocity_covariance, dtype=float).reshape(3, 3)

    if not (
        np.all(np.isfinite(position))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(covariance))
    ):
        raise ValueError("prediction inputs must be finite")

    dt = float(prediction_dt_sec)
    center = position[:2] + velocity[:2] * dt

    covariance_xy = covariance[:2, :2]
    covariance_xy = 0.5 * (covariance_xy + covariance_xy.T)
    position_covariance_xy = covariance_xy * (dt**2)

    eigenvalues, eigenvectors = np.linalg.eigh(position_covariance_xy)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    radii = float(sigma_scale) * np.sqrt(eigenvalues)

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        int(point_count),
        endpoint=False,
    )
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = center[:, None] + (
        eigenvectors @ np.diag(radii) @ unit_circle
    )
    return center, ellipse.T
