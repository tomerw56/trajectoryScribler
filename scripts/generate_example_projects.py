from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from trajectory_app.core import (
    build_trajectory,
    sample_motion,
    sample_velocity_covariance,
    terrain_surface_position,
)
from trajectory_app.covariance_policy import CovarianceScope
from trajectory_app.covariance_stability import CovarianceStabilityCalculator
from trajectory_app.models import (
    AltitudeMode,
    MotionConfig,
    TerrainConfig,
    TrajectoryConfig,
    dataclass_dict,
)
from trajectory_app.terrain_io import load_terrain, save_terrain


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TERRAINS = ROOT / "terrains"


def arc_points(
    center_x: float,
    center_y: float,
    radius_m: float,
    start_deg: float,
    end_deg: float,
    count: int,
) -> list[list[float]]:
    angles = np.radians(np.linspace(start_deg, end_deg, count))
    return np.column_stack(
        (
            center_x + radius_m * np.cos(angles),
            center_y + radius_m * np.sin(angles),
        )
    ).tolist()


def target(
    target_id: str,
    name: str,
    color: str,
    points: list[list[float]],
    *,
    speed_mps: float,
    covariance_window_sec: float = 5.0,
    stability_window_sec: float = 6.0,
    prediction_dt_sec: float = 2.0,
    clearance_m: float = 5.0,
    altitude_mode: AltitudeMode = AltitudeMode.TERRAIN_CLEARANCE,
    cruise_altitude_m: float = 25.0,
) -> dict:
    trajectory_config = TrajectoryConfig(
        path_resolution_m=0.25,
        altitude_mode=altitude_mode,
        clearance_m=clearance_m,
        cruise_altitude_m=cruise_altitude_m,
        max_climb_angle_deg=18.0,
        smoothing_passes=2,
    )
    motion_config = MotionConfig(
        speed_mps=float(speed_mps),
        sample_dt_sec=0.25,
        covariance_sample_rate_sec=0.5,
        covariance_window_sec=float(covariance_window_sec),
        prediction_dt_sec=float(prediction_dt_sec),
        covariance_stability_window_sec=float(stability_window_sec),
    )
    return {
        "id": target_id,
        "name": name,
        "color": color,
        "scribble_xy": [[float(x), float(y)] for x, y in points],
        "trajectory_config": dataclass_dict(trajectory_config),
        "motion_config": dataclass_dict(motion_config),
    }


def cameraman(
    cam_id: str,
    name: str,
    color: str,
    terrain: np.ndarray,
    terrain_cfg: TerrainConfig,
    x: float,
    y: float,
    *,
    radius: float,
    start_deg: float,
    end_deg: float,
) -> dict:
    xyz = terrain_surface_position(terrain, terrain_cfg, x, y)
    return {
        "id": cam_id,
        "name": name,
        "color": color,
        "position_xyz": [float(v) for v in xyz],
        "view_radius_m": float(radius),
        "view_start_angle_deg": float(start_deg),
        "view_end_angle_deg": float(end_deg),
    }


def write_project(
    path: Path,
    terrain_path: Path,
    scope: CovarianceScope,
    targets: list[dict],
    cameramen: list[dict],
) -> None:
    payload = {
        "version": 6,
        "terrain_file": str(
            Path("..") / "terrains" / terrain_path.name
        ).replace("\\", "/"),
        "covariance_scope": scope.value,
        "targets": targets,
        "cameramen": cameramen,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_flat_showcase() -> tuple[Path, list[dict]]:
    flat_cfg = TerrainConfig(
        x_min=-100.0,
        x_max=100.0,
        y_min=-75.0,
        y_max=75.0,
        grid_size=121,
        seed=808,
        base_height=10.0,
        valley_count=0,
        mountain_count=0,
        valley_depth_min=0.0,
        valley_depth_max=0.0,
        mountain_height_min=0.0,
        mountain_height_max=0.0,
        min_sigma=2.0,
        max_sigma=7.0,
        noise_amplitude=0.0,
    )
    flat = np.full((flat_cfg.grid_size, flat_cfg.grid_size), 10.0)
    flat_path = save_terrain(TERRAINS / "stability_demo_flat.terrain", flat_cfg, flat)

    targets = [
        target(
            "demo_straight",
            "Straight — tiny covariance",
            "#46d36f",
            [[-85, -55], [85, -55]],
            speed_mps=3.0,
        ),
        target(
            "demo_near_straight",
            "Near-straight — small covariance",
            "#4aa8ff",
            [
                [-90, -25],
                [-55, -24.8],
                [-20, -25.2],
                [20, -24.9],
                [55, -25.15],
                [90, -25],
            ],
            speed_mps=3.5,
        ),
        target(
            "demo_gentle_arc",
            "Gentle arc — moderate covariance",
            "#ffd166",
            arc_points(0, -10, 75, 200, 340, 20),
            speed_mps=4.0,
        ),
        target(
            "demo_tight_arc",
            "Tight arc — sustained changing",
            "#ff9f43",
            arc_points(-35, 20, 20, -80, 190, 24),
            speed_mps=5.5,
        ),
        target(
            "demo_s_curve",
            "S-curve — large covariance",
            "#e66cff",
            [
                [-90, 5],
                [-70, 18],
                [-50, 28],
                [-25, 25],
                [0, 10],
                [25, -5],
                [50, -8],
                [70, 2],
                [90, 18],
            ],
            speed_mps=5.0,
        ),
        target(
            "demo_zigzag",
            "Zigzag — very large covariance",
            "#ff5c5c",
            [
                [-90, 45],
                [-65, 60],
                [-40, 30],
                [-15, 60],
                [10, 28],
                [35, 60],
                [60, 30],
                [90, 50],
            ],
            speed_mps=7.0,
        ),
    ]

    cameras = [
        cameraman(
            "cam_west",
            "West observer",
            "#8bd3dd",
            flat,
            flat_cfg,
            -72,
            -8,
            radius=65,
            start_deg=-45,
            end_deg=45,
        ),
        cameraman(
            "cam_south",
            "South observer",
            "#b8f2a1",
            flat,
            flat_cfg,
            0,
            -62,
            radius=70,
            start_deg=45,
            end_deg=135,
        ),
        cameraman(
            "cam_north",
            "North observer",
            "#f6bd60",
            flat,
            flat_cfg,
            0,
            63,
            radius=70,
            start_deg=225,
            end_deg=315,
        ),
        cameraman(
            "cam_east",
            "East observer",
            "#cdb4db",
            flat,
            flat_cfg,
            72,
            4,
            radius=65,
            start_deg=135,
            end_deg=225,
        ),
        cameraman(
            "cam_center",
            "Center observer",
            "#ffffff",
            flat,
            flat_cfg,
            0,
            0,
            radius=48,
            start_deg=-70,
            end_deg=70,
        ),
    ]

    path = EXAMPLES / "01_covariance_showcase_flat.trajectory"
    write_project(
        path,
        flat_path,
        CovarianceScope.PREDICTION_XY,
        targets,
        cameras,
    )
    return path, targets


def build_basic_patrol() -> Path:
    flat_cfg, flat = load_terrain(TERRAINS / "stability_demo_flat.terrain")
    targets = [
        target(
            "patrol_horizontal",
            "Horizontal patrol",
            "#46d36f",
            [[-85, -45], [85, -45]],
            speed_mps=3.0,
        ),
        target(
            "patrol_diagonal",
            "Diagonal patrol",
            "#4aa8ff",
            [[-80, -10], [80, 50]],
            speed_mps=3.5,
        ),
        target(
            "patrol_arc",
            "Curved patrol",
            "#ffd166",
            arc_points(0, 0, 52, 205, 335, 16),
            speed_mps=4.0,
        ),
        target(
            "patrol_turns",
            "Turning patrol",
            "#ff6b6b",
            [[-85, 45], [-35, 45], [-35, 5], [25, 5], [25, 45], [85, 45]],
            speed_mps=5.0,
        ),
    ]
    cameras = [
        cameraman(
            "patrol_cam_w",
            "Patrol west",
            "#8bd3dd",
            flat,
            flat_cfg,
            -65,
            0,
            radius=60,
            start_deg=-55,
            end_deg=55,
        ),
        cameraman(
            "patrol_cam_c",
            "Patrol center",
            "#ffffff",
            flat,
            flat_cfg,
            0,
            5,
            radius=50,
            start_deg=20,
            end_deg=160,
        ),
        cameraman(
            "patrol_cam_e",
            "Patrol east",
            "#cdb4db",
            flat,
            flat_cfg,
            65,
            25,
            radius=60,
            start_deg=125,
            end_deg=235,
        ),
    ]
    path = EXAMPLES / "02_basic_multi_target_patrol.trajectory"
    write_project(
        path,
        TERRAINS / "stability_demo_flat.terrain",
        CovarianceScope.PREDICTION_XY,
        targets,
        cameras,
    )
    return path


def build_full_3d_hills() -> Path:
    terrain_path = TERRAINS / "multi_hills.terrain"
    terrain_cfg, terrain = load_terrain(terrain_path)

    targets = [
        target(
            "hills_straight",
            "Hill crossing — straight XY",
            "#46d36f",
            [[-90, -48], [90, -48]],
            speed_mps=4.0,
            clearance_m=6.0,
        ),
        target(
            "hills_diagonal",
            "Hill diagonal — mixed XYZ covariance",
            "#4aa8ff",
            [[-88, -25], [-50, -5], [-5, 18], [45, 32], [88, 45]],
            speed_mps=4.5,
            clearance_m=6.0,
        ),
        target(
            "hills_s",
            "Hill S-curve — large 3D covariance",
            "#e66cff",
            [
                [-90, 48],
                [-65, 25],
                [-35, 45],
                [-5, 10],
                [25, 38],
                [55, 5],
                [88, 28],
            ],
            speed_mps=5.5,
            clearance_m=6.0,
        ),
        target(
            "hills_cruise",
            "Cruise-altitude route",
            "#ffd166",
            [[-88, 5], [-50, 8], [-5, 6], [40, 10], [88, 8]],
            speed_mps=4.0,
            clearance_m=5.0,
            altitude_mode=AltitudeMode.CRUISE_ALTITUDE,
            cruise_altitude_m=55.0,
        ),
    ]

    cameras = [
        cameraman(
            "hill_cam_sw",
            "South-west hill observer",
            "#8bd3dd",
            terrain,
            terrain_cfg,
            -72,
            -52,
            radius=70,
            start_deg=-20,
            end_deg=75,
        ),
        cameraman(
            "hill_cam_nw",
            "North-west hill observer",
            "#b8f2a1",
            terrain,
            terrain_cfg,
            -45,
            52,
            radius=65,
            start_deg=-75,
            end_deg=25,
        ),
        cameraman(
            "hill_cam_center",
            "Central hill observer",
            "#ffffff",
            terrain,
            terrain_cfg,
            0,
            0,
            radius=55,
            start_deg=15,
            end_deg=165,
        ),
        cameraman(
            "hill_cam_e",
            "East hill observer",
            "#f6bd60",
            terrain,
            terrain_cfg,
            68,
            12,
            radius=70,
            start_deg=125,
            end_deg=235,
        ),
    ]

    path = EXAMPLES / "03_full_3d_hills_showcase.trajectory"
    write_project(
        path,
        terrain_path,
        CovarianceScope.FULL_3D,
        targets,
        cameras,
    )
    return path



def _terrain(name: str) -> tuple[Path, TerrainConfig, np.ndarray]:
    path = TERRAINS / name
    config, heights = load_terrain(path)
    return path, config, heights


def build_canyon_crossings() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("canyon.terrain")

    targets = [
        target(
            "canyon_low_crossing",
            "Canyon low crossing",
            "#46d36f",
            [[-92, -48], [-50, -42], [0, -38], [48, -43], [92, -47]],
            speed_mps=3.8,
            clearance_m=6.0,
        ),
        target(
            "canyon_diagonal",
            "Canyon diagonal",
            "#4aa8ff",
            [[-90, -25], [-55, -5], [-10, 12], [38, 30], [90, 52]],
            speed_mps=4.5,
            clearance_m=7.0,
        ),
        target(
            "canyon_s_curve",
            "Canyon S-curve",
            "#e66cff",
            [[-90, 28], [-62, 50], [-28, 22], [5, 48], [35, 18], [62, 42], [90, 20]],
            speed_mps=5.0,
            clearance_m=6.0,
        ),
        target(
            "canyon_hard_turn",
            "Canyon hard turn",
            "#ff6b6b",
            [[-88, 58], [-35, 58], [-35, 8], [18, 8], [18, 52], [88, 52]],
            speed_mps=5.8,
            clearance_m=6.0,
        ),
        target(
            "canyon_cruise",
            "Canyon cruise-altitude bird",
            "#ffd166",
            [[-92, 3], [-50, 0], [-5, 3], [42, 0], [92, 4]],
            speed_mps=4.2,
            clearance_m=5.0,
            altitude_mode=AltitudeMode.CRUISE_ALTITUDE,
            cruise_altitude_m=58.0,
        ),
    ]

    cameras = [
        cameraman("canyon_cam_w", "Canyon west", "#8bd3dd", terrain, terrain_cfg, -78, -10,
                  radius=72, start_deg=-45, end_deg=55),
        cameraman("canyon_cam_sw", "Canyon south-west", "#b8f2a1", terrain, terrain_cfg, -35, -62,
                  radius=68, start_deg=30, end_deg=120),
        cameraman("canyon_cam_center", "Canyon center", "#ffffff", terrain, terrain_cfg, 0, 0,
                  radius=58, start_deg=-25, end_deg=155),
        cameraman("canyon_cam_ne", "Canyon north-east", "#f6bd60", terrain, terrain_cfg, 45, 58,
                  radius=68, start_deg=205, end_deg=305),
        cameraman("canyon_cam_e", "Canyon east", "#cdb4db", terrain, terrain_cfg, 80, 2,
                  radius=72, start_deg=125, end_deg=225),
    ]

    path = EXAMPLES / "04_canyon_crossings.trajectory"
    write_project(path, terrain_path, CovarianceScope.FULL_3D, targets, cameras)
    return path


def build_ridge_pass_watch() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("ridge_pass.terrain")

    targets = [
        target(
            "ridge_pass_west_east",
            "Pass west-to-east",
            "#46d36f",
            [[-92, -8], [-55, -5], [-20, 0], [20, 2], [55, 4], [92, 6]],
            speed_mps=4.0,
            clearance_m=6.0,
        ),
        target(
            "ridge_pass_east_west",
            "Pass east-to-west",
            "#4aa8ff",
            [[92, 18], [52, 16], [12, 10], [-30, 8], [-65, 4], [-92, 0]],
            speed_mps=4.5,
            clearance_m=6.0,
        ),
        target(
            "ridge_northern_arc",
            "Northern ridge arc",
            "#ffd166",
            arc_points(0, 8, 74, 205, 335, 20),
            speed_mps=4.3,
            clearance_m=7.0,
        ),
        target(
            "ridge_southern_s",
            "Southern pass S-curve",
            "#e66cff",
            [[-90, -52], [-62, -30], [-32, -50], [0, -24], [34, -48], [64, -26], [92, -44]],
            speed_mps=5.2,
            clearance_m=6.0,
        ),
        target(
            "ridge_turner",
            "Pass turning bird",
            "#ff6b6b",
            [[-82, 48], [-35, 48], [-35, 14], [18, 14], [18, 48], [82, 48]],
            speed_mps=5.8,
            clearance_m=6.0,
        ),
        target(
            "ridge_cruise",
            "High cruise over pass",
            "#9b8cff",
            [[-92, 28], [-48, 26], [0, 24], [48, 28], [92, 24]],
            speed_mps=4.8,
            clearance_m=5.0,
            altitude_mode=AltitudeMode.CRUISE_ALTITUDE,
            cruise_altitude_m=72.0,
        ),
    ]

    cameras = [
        cameraman("ridge_cam_w", "Pass west observer", "#8bd3dd", terrain, terrain_cfg, -78, 0,
                  radius=65, start_deg=-40, end_deg=45),
        cameraman("ridge_cam_s", "Pass south observer", "#b8f2a1", terrain, terrain_cfg, -12, -60,
                  radius=72, start_deg=35, end_deg=135),
        cameraman("ridge_cam_mid", "Pass middle observer", "#ffffff", terrain, terrain_cfg, 0, 6,
                  radius=52, start_deg=-80, end_deg=100),
        cameraman("ridge_cam_n", "Pass north observer", "#f6bd60", terrain, terrain_cfg, 18, 60,
                  radius=70, start_deg=215, end_deg=320),
        cameraman("ridge_cam_e", "Pass east observer", "#cdb4db", terrain, terrain_cfg, 80, 8,
                  radius=65, start_deg=135, end_deg=225),
    ]

    path = EXAMPLES / "05_ridge_pass_watch.trajectory"
    write_project(path, terrain_path, CovarianceScope.FULL_3D, targets, cameras)
    return path


def build_big_top_orbits() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("big_top.terrain")

    targets = [
        target(
            "big_top_outer_arc",
            "Outer mountain arc",
            "#46d36f",
            arc_points(0, 0, 68, 185, 355, 28),
            speed_mps=4.0,
            clearance_m=8.0,
        ),
        target(
            "big_top_inner_arc",
            "Inner mountain arc",
            "#4aa8ff",
            arc_points(0, 0, 42, 15, 210, 28),
            speed_mps=4.8,
            clearance_m=8.0,
        ),
        target(
            "big_top_crossing",
            "Summit crossing",
            "#ffd166",
            [[-92, -5], [-48, -3], [-15, 0], [15, 0], [48, 3], [92, 5]],
            speed_mps=4.2,
            clearance_m=8.0,
        ),
        target(
            "big_top_s",
            "Mountain S-route",
            "#e66cff",
            [[-88, -50], [-58, -28], [-35, -52], [-5, -20], [25, -48], [55, -22], [88, -42]],
            speed_mps=5.4,
            clearance_m=8.0,
        ),
        target(
            "big_top_fast_turns",
            "Fast summit turns",
            "#ff6b6b",
            [[-82, 48], [-42, 50], [-42, 12], [0, 12], [0, 52], [42, 52], [42, 14], [82, 16]],
            speed_mps=6.2,
            clearance_m=8.0,
        ),
        target(
            "big_top_cruise",
            "High cruise orbit",
            "#9b8cff",
            arc_points(0, 0, 78, 200, 340, 22),
            speed_mps=5.0,
            clearance_m=5.0,
            altitude_mode=AltitudeMode.CRUISE_ALTITUDE,
            cruise_altitude_m=118.0,
        ),
    ]

    cameras = [
        cameraman("top_cam_w", "Mountain west", "#8bd3dd", terrain, terrain_cfg, -82, 0,
                  radius=75, start_deg=-50, end_deg=45),
        cameraman("top_cam_sw", "Mountain south-west", "#b8f2a1", terrain, terrain_cfg, -40, -60,
                  radius=72, start_deg=15, end_deg=115),
        cameraman("top_cam_s", "Mountain south", "#ffffff", terrain, terrain_cfg, 18, -65,
                  radius=68, start_deg=50, end_deg=135),
        cameraman("top_cam_ne", "Mountain north-east", "#f6bd60", terrain, terrain_cfg, 45, 58,
                  radius=72, start_deg=195, end_deg=300),
        cameraman("top_cam_e", "Mountain east", "#cdb4db", terrain, terrain_cfg, 82, 5,
                  radius=75, start_deg=135, end_deg=230),
    ]

    path = EXAMPLES / "06_big_top_orbits.trajectory"
    write_project(path, terrain_path, CovarianceScope.FULL_3D, targets, cameras)
    return path


def build_multi_hills_mixed_flock() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("multi_hills.terrain")

    targets = [
        target("flock_1", "Flock A — straight", "#46d36f", [[-92, -58], [92, -58]], speed_mps=3.2, clearance_m=6),
        target("flock_2", "Flock B — diagonal", "#4aa8ff", [[-90, -35], [-45, -12], [0, 8], [45, 28], [90, 48]], speed_mps=4.0, clearance_m=6),
        target("flock_3", "Flock C — broad arc", "#ffd166", arc_points(0, 0, 70, 195, 340, 22), speed_mps=4.5, clearance_m=6),
        target("flock_4", "Flock D — S route", "#e66cff", [[-90, 8], [-65, 28], [-35, 5], [-5, 30], [25, 4], [55, 28], [90, 8]], speed_mps=5.0, clearance_m=6),
        target("flock_5", "Flock E — repeated turns", "#ff6b6b", [[-88, 52], [-58, 52], [-58, 20], [-25, 20], [-25, 52], [10, 52], [10, 18], [45, 18], [45, 50], [88, 50]], speed_mps=6.0, clearance_m=6),
        target("flock_6", "Flock F — low wander", "#7bdff2", [[-90, -20], [-60, -19], [-30, -22], [0, -20], [30, -18], [60, -21], [90, -20]], speed_mps=3.6, clearance_m=6),
        target("flock_7", "Flock G — cruise", "#9b8cff", [[-90, 38], [-45, 35], [0, 40], [45, 36], [90, 40]], speed_mps=4.4, clearance_m=5, altitude_mode=AltitudeMode.CRUISE_ALTITUDE, cruise_altitude_m=62),
    ]

    cameras = [
        cameraman("mh_cam_w", "Hill west", "#8bd3dd", terrain, terrain_cfg, -82, -8, radius=70, start_deg=-55, end_deg=50),
        cameraman("mh_cam_sw", "Hill south-west", "#b8f2a1", terrain, terrain_cfg, -45, -58, radius=66, start_deg=20, end_deg=115),
        cameraman("mh_cam_c", "Hill center", "#ffffff", terrain, terrain_cfg, 0, 0, radius=52, start_deg=-70, end_deg=115),
        cameraman("mh_cam_n", "Hill north", "#f6bd60", terrain, terrain_cfg, 0, 62, radius=66, start_deg=215, end_deg=320),
        cameraman("mh_cam_se", "Hill south-east", "#ffafcc", terrain, terrain_cfg, 45, -55, radius=66, start_deg=65, end_deg=160),
        cameraman("mh_cam_e", "Hill east", "#cdb4db", terrain, terrain_cfg, 82, 8, radius=70, start_deg=130, end_deg=235),
    ]

    path = EXAMPLES / "07_multi_hills_mixed_flock.trajectory"
    write_project(path, terrain_path, CovarianceScope.PREDICTION_XY, targets, cameras)
    return path


def build_dense_camera_network_flat() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("stability_demo_flat.terrain")

    targets = [
        target("dense_bird_1", "Dense bird 1 — west/east", "#46d36f", [[-92, -55], [92, -55]], speed_mps=3.0),
        target("dense_bird_2", "Dense bird 2 — diagonal up", "#4aa8ff", [[-92, -42], [92, 42]], speed_mps=3.8),
        target("dense_bird_3", "Dense bird 3 — diagonal down", "#ffd166", [[-92, 42], [92, -42]], speed_mps=4.0),
        target("dense_bird_4", "Dense bird 4 — broad arc", "#e66cff", arc_points(0, 0, 64, 195, 345, 22), speed_mps=4.3),
        target("dense_bird_5", "Dense bird 5 — S", "#ff6b6b", [[-90, 8], [-60, 28], [-30, 5], [0, 30], [30, 4], [60, 27], [90, 8]], speed_mps=5.0),
        target("dense_bird_6", "Dense bird 6 — box turns", "#7bdff2", [[-75, 55], [-25, 55], [-25, 18], [25, 18], [25, 55], [75, 55]], speed_mps=5.6),
        target("dense_bird_7", "Dense bird 7 — low wander", "#b8f2a1", [[-90, -15], [-55, -14], [-20, -17], [20, -14], [55, -16], [90, -15]], speed_mps=3.5),
        target("dense_bird_8", "Dense bird 8 — tight arc", "#ff9f43", arc_points(15, -5, 28, -100, 185, 26), speed_mps=5.4),
    ]

    camera_specs = [
        ("dense_cam_w", "Dense west", "#8bd3dd", -82, 0, -45, 45),
        ("dense_cam_nw", "Dense north-west", "#b8f2a1", -55, 55, -70, 20),
        ("dense_cam_n", "Dense north", "#f6bd60", 0, 65, 220, 320),
        ("dense_cam_ne", "Dense north-east", "#ffafcc", 55, 55, 160, 250),
        ("dense_cam_e", "Dense east", "#cdb4db", 82, 0, 135, 225),
        ("dense_cam_se", "Dense south-east", "#9b8cff", 55, -55, 110, 200),
        ("dense_cam_s", "Dense south", "#ffffff", 0, -65, 40, 140),
        ("dense_cam_sw", "Dense south-west", "#7bdff2", -55, -55, -20, 75),
    ]
    cameras = [
        cameraman(cid, name, color, terrain, terrain_cfg, x, y, radius=52, start_deg=a0, end_deg=a1)
        for cid, name, color, x, y, a0, a1 in camera_specs
    ]

    path = EXAMPLES / "08_dense_camera_network_flat.trajectory"
    write_project(path, terrain_path, CovarianceScope.PREDICTION_XY, targets, cameras)
    return path


def build_sparse_camera_challenge() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("canyon.terrain")

    targets = [
        target("sparse_bird_1", "Sparse challenge straight", "#46d36f", [[-92, -55], [92, -55]], speed_mps=3.2, clearance_m=6),
        target("sparse_bird_2", "Sparse challenge diagonal", "#4aa8ff", [[-92, -35], [92, 52]], speed_mps=4.0, clearance_m=6),
        target("sparse_bird_3", "Sparse challenge arc", "#ffd166", arc_points(0, 0, 72, 200, 340, 20), speed_mps=4.4, clearance_m=6),
        target("sparse_bird_4", "Sparse challenge S", "#e66cff", [[-90, 20], [-55, 48], [-20, 16], [15, 48], [50, 15], [90, 40]], speed_mps=5.2, clearance_m=6),
        target("sparse_bird_5", "Sparse challenge turns", "#ff6b6b", [[-85, 55], [-35, 55], [-35, 0], [20, 0], [20, 48], [85, 48]], speed_mps=5.8, clearance_m=6),
        target("sparse_bird_6", "Sparse challenge cruise", "#9b8cff", [[-92, 0], [-45, 5], [0, 0], [45, 5], [92, 0]], speed_mps=4.6, clearance_m=5, altitude_mode=AltitudeMode.CRUISE_ALTITUDE, cruise_altitude_m=62),
    ]

    cameras = [
        cameraman("sparse_cam_w", "Sparse west", "#8bd3dd", terrain, terrain_cfg, -78, -5, radius=58, start_deg=-35, end_deg=55),
        cameraman("sparse_cam_center", "Sparse center", "#ffffff", terrain, terrain_cfg, 0, -8, radius=48, start_deg=5, end_deg=135),
        cameraman("sparse_cam_e", "Sparse east", "#cdb4db", terrain, terrain_cfg, 78, 10, radius=58, start_deg=130, end_deg=225),
    ]

    path = EXAMPLES / "09_sparse_camera_challenge_canyon.trajectory"
    write_project(path, terrain_path, CovarianceScope.PREDICTION_XY, targets, cameras)
    return path


def build_clearance_vs_cruise() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("big_top.terrain")

    route_a = [[-92, -28], [-45, -25], [0, -22], [45, -25], [92, -28]]
    route_b = [[-88, 8], [-52, 28], [-15, 8], [20, 30], [55, 10], [88, 26]]
    route_c = arc_points(0, 0, 64, 195, 345, 22)

    targets = [
        target("pair_a_clearance", "Route A — terrain clearance", "#46d36f", route_a, speed_mps=4.0, clearance_m=7),
        target("pair_a_cruise", "Route A — cruise altitude", "#7bdff2", route_a, speed_mps=4.0, clearance_m=5, altitude_mode=AltitudeMode.CRUISE_ALTITUDE, cruise_altitude_m=115),
        target("pair_b_clearance", "Route B — terrain clearance", "#ffd166", route_b, speed_mps=4.8, clearance_m=7),
        target("pair_b_cruise", "Route B — cruise altitude", "#ff9f43", route_b, speed_mps=4.8, clearance_m=5, altitude_mode=AltitudeMode.CRUISE_ALTITUDE, cruise_altitude_m=118),
        target("pair_c_clearance", "Route C — terrain clearance", "#e66cff", route_c, speed_mps=5.0, clearance_m=7),
        target("pair_c_cruise", "Route C — cruise altitude", "#9b8cff", route_c, speed_mps=5.0, clearance_m=5, altitude_mode=AltitudeMode.CRUISE_ALTITUDE, cruise_altitude_m=120),
    ]

    cameras = [
        cameraman("pair_cam_w", "Pair west", "#8bd3dd", terrain, terrain_cfg, -80, 0, radius=72, start_deg=-45, end_deg=50),
        cameraman("pair_cam_s", "Pair south", "#b8f2a1", terrain, terrain_cfg, 0, -62, radius=72, start_deg=40, end_deg=140),
        cameraman("pair_cam_n", "Pair north", "#f6bd60", terrain, terrain_cfg, 0, 62, radius=72, start_deg=220, end_deg=320),
        cameraman("pair_cam_e", "Pair east", "#cdb4db", terrain, terrain_cfg, 80, 5, radius=72, start_deg=130, end_deg=230),
    ]

    path = EXAMPLES / "10_clearance_vs_cruise_big_top.trajectory"
    write_project(path, terrain_path, CovarianceScope.FULL_3D, targets, cameras)
    return path


def build_high_variance_maneuvers() -> Path:
    terrain_path, terrain_cfg, terrain = _terrain("stability_demo_flat.terrain")

    targets = [
        target("hv_baseline", "Baseline straight", "#46d36f", [[-92, -58], [92, -58]], speed_mps=3.0),
        target("hv_saw_1", "Sawtooth fast", "#ff6b6b", [[-90, -35], [-65, -10], [-40, -38], [-15, -8], [10, -40], [35, -5], [60, -36], [90, -10]], speed_mps=7.0),
        target("hv_saw_2", "Sawtooth tighter", "#ff9f43", [[-88, 0], [-68, 22], [-48, -4], [-28, 24], [-8, -6], [12, 24], [32, -4], [52, 22], [72, -2], [90, 18]], speed_mps=6.5),
        target("hv_box", "Repeated box turns", "#e66cff", [[-80, 52], [-45, 52], [-45, 18], [-10, 18], [-10, 52], [25, 52], [25, 18], [60, 18], [60, 52], [88, 52]], speed_mps=6.0),
        target("hv_tight_arc", "Very tight arc", "#4aa8ff", arc_points(-25, 20, 18, -100, 230, 30), speed_mps=6.0),
        target("hv_reverse_s", "Reverse S maneuver", "#ffd166", [[-90, 30], [-62, 50], [-35, 22], [-8, 48], [20, 18], [48, 45], [90, 20]], speed_mps=5.8),
        target("hv_wander", "Moderate wander control", "#7bdff2", [[-90, -15], [-55, -10], [-20, -17], [15, -9], [50, -16], [90, -12]], speed_mps=4.0),
    ]

    cameras = [
        cameraman("hv_cam_w", "Variance west", "#8bd3dd", terrain, terrain_cfg, -82, 0, radius=65, start_deg=-50, end_deg=50),
        cameraman("hv_cam_sw", "Variance south-west", "#b8f2a1", terrain, terrain_cfg, -45, -58, radius=65, start_deg=20, end_deg=115),
        cameraman("hv_cam_n", "Variance north", "#f6bd60", terrain, terrain_cfg, 0, 63, radius=65, start_deg=220, end_deg=320),
        cameraman("hv_cam_c", "Variance center", "#ffffff", terrain, terrain_cfg, 0, 0, radius=48, start_deg=-90, end_deg=90),
        cameraman("hv_cam_se", "Variance south-east", "#ffafcc", terrain, terrain_cfg, 45, -58, radius=65, start_deg=65, end_deg=160),
        cameraman("hv_cam_e", "Variance east", "#cdb4db", terrain, terrain_cfg, 82, 0, radius=65, start_deg=130, end_deg=230),
    ]

    path = EXAMPLES / "11_high_variance_maneuvers_flat.trajectory"
    write_project(path, terrain_path, CovarianceScope.PREDICTION_XY, targets, cameras)
    return path


def summarize_showcase(project_path: Path, target_payloads: list[dict]) -> str:
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    terrain_path = (project_path.parent / payload["terrain_file"]).resolve()
    terrain_cfg, terrain = load_terrain(terrain_path)
    calculator = CovarianceStabilityCalculator()

    rows = []
    for item in target_payloads:
        tc = item["trajectory_config"]
        mc = item["motion_config"]
        trajectory_cfg = TrajectoryConfig(
            path_resolution_m=tc["path_resolution_m"],
            altitude_mode=AltitudeMode(tc["altitude_mode"]),
            clearance_m=tc["clearance_m"],
            cruise_altitude_m=tc["cruise_altitude_m"],
            max_climb_angle_deg=tc["max_climb_angle_deg"],
            smoothing_passes=tc["smoothing_passes"],
        )
        motion_cfg = MotionConfig(**mc)
        trajectory = build_trajectory(
            item["scribble_xy"],
            terrain,
            terrain_cfg,
            trajectory_cfg,
        )
        samples = sample_motion(trajectory, terrain, terrain_cfg, motion_cfg)
        covariance = sample_velocity_covariance(samples, motion_cfg)
        xy_trace = covariance.cxx + covariance.cyy
        result = calculator.evaluate(
            covariance,
            end_time_sec=float(covariance.time_sec[-1]),
            source_warmup_sec=motion_cfg.covariance_window_sec,
            scope=CovarianceScope.PREDICTION_XY,
        )
        rows.append(
            (
                item["name"],
                float(samples.time_sec[-1]),
                float(np.max(xy_trace)),
                float(np.mean(xy_trace)),
                result.state.value,
            )
        )

    lines = [
        "# Example trajectory projects",
        "",
        "The `examples/` folder contains ready-to-load `.trajectory` projects.",
        "",
        "## 01 — covariance showcase, flat terrain",
        "",
        "Six targets and five cameramen. This is the easiest project for comparing",
        "small, moderate, and very large XY velocity covariance because terrain Z",
        "does not contribute.",
        "",
        "| Target | Duration (s) | Max XY trace | Mean XY trace | End stability |",
        "|---|---:|---:|---:|---|",
    ]
    for name, duration, max_trace, mean_trace, state in rows:
        lines.append(
            f"| {name} | {duration:.1f} | {max_trace:.4f} | "
            f"{mean_trace:.4f} | {state.upper()} |"
        )

    lines += [
        "",
        "The end-state is only a snapshot; during playback a target may transition",
        "between STABLE, CHANGING, and UNSTABLE as the recent 6-second covariance",
        "history changes.",
        "",
        "## 02 — basic multi-target patrol",
        "",
        "A simpler four-target / three-cameraman project for ordinary UI testing.",
        "",
        "## 03 — full 3D hills showcase",
        "",
        "Four targets and four cameramen on `multi_hills.terrain`, with system",
        "covariance scope preset to `FULL_3D`. Use this to see the difference",
        "between XY-only interpretation and complete XYZ velocity covariance.",
        "",
        "## 04 — canyon crossings",
        "",
        "Five birds and five cameramen on `canyon.terrain`. Mixed straight, diagonal,",
        "S-curve, hard-turn, and cruise-altitude routes. Preset to Full 3D.",
        "",
        "## 05 — ridge-pass watch",
        "",
        "Six birds and five cameramen on `ridge_pass.terrain`, including opposing",
        "pass crossings, arcs, sharp turns, and a high cruise route. Preset to Full 3D.",
        "",
        "## 06 — big-top orbits",
        "",
        "Six birds and five cameramen around `big_top.terrain`: inner/outer arcs,",
        "summit crossing, S-route, fast turns, and a high cruise orbit.",
        "",
        "## 07 — multi-hills mixed flock",
        "",
        "Seven birds and six cameramen on `multi_hills.terrain`; useful for ordinary",
        "Prediction-XY playback with low, moderate, and aggressive trajectories.",
        "",
        "## 08 — dense camera network",
        "",
        "Eight birds and eight cameramen on flat terrain. Use this to stress the UI",
        "with many simultaneous paths, sectors, estimates, and covariance envelopes.",
        "",
        "## 09 — sparse camera challenge",
        "",
        "Six birds but only three cameramen on canyon terrain. The geometry is",
        "intentionally sparse and is useful for observer-layout experiments later.",
        "",
        "## 10 — clearance vs cruise",
        "",
        "Three route pairs over `big_top.terrain`. Each pair uses the same XY route",
        "once in terrain-clearance mode and once in cruise-altitude mode. Full 3D.",
        "",
        "## 11 — high-variance maneuvers",
        "",
        "Seven birds and six cameramen on flat terrain. Includes a straight baseline",
        "plus sawtooths, box turns, a tight arc, and aggressive S maneuvers.",
        "",
        "## Regeneration",
        "",
        "From the repository root:",
        "",
        "```powershell",
        "python .\\scripts\\generate_example_projects.py",
        "```",
        "",
        "The generator is deterministic.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    TERRAINS.mkdir(parents=True, exist_ok=True)

    showcase_path, showcase_targets = build_flat_showcase()
    generated = [
        showcase_path,
        build_basic_patrol(),
        build_full_3d_hills(),
        build_canyon_crossings(),
        build_ridge_pass_watch(),
        build_big_top_orbits(),
        build_multi_hills_mixed_flock(),
        build_dense_camera_network_flat(),
        build_sparse_camera_challenge(),
        build_clearance_vs_cruise(),
        build_high_variance_maneuvers(),
    ]

    (EXAMPLES / "README.md").write_text(
        summarize_showcase(showcase_path, showcase_targets),
        encoding="utf-8",
    )

    manifest = []
    for project_path in generated:
        payload = json.loads(project_path.read_text(encoding="utf-8"))
        manifest.append(
            {
                "project": project_path.name,
                "terrain": Path(payload["terrain_file"]).name,
                "covariance_scope": payload["covariance_scope"],
                "bird_count": len(payload["targets"]),
                "cameraman_count": len(payload["cameramen"]),
                "birds": [item["name"] for item in payload["targets"]],
                "cameramen": [item["name"] for item in payload["cameramen"]],
            }
        )
    (EXAMPLES / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Generated:")
    for path in generated:
        print(" ", path.relative_to(ROOT))
    print(" ", (TERRAINS / "stability_demo_flat.terrain").relative_to(ROOT))
    print(" ", (EXAMPLES / "README.md").relative_to(ROOT))
    print(" ", (EXAMPLES / "scenario_manifest.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
