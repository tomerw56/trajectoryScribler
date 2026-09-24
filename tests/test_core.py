import numpy as np
from trajectory_app.core import generate_terrain, build_trajectory, sample_motion, sample_velocity_covariance
from trajectory_app.models import TerrainConfig, TrajectoryConfig, MotionConfig, AltitudeMode
from trajectory_app.terrain_io import save_terrain, load_terrain


def make():
    tc=TerrainConfig(seed=4,grid_size=80,x_min=-15,x_max=25,y_min=-15,y_max=15)
    terrain=generate_terrain(tc)
    scribble=[[-14,-13],[-5,-5],[3,1],[12,9],[23,14]]
    return tc,terrain,scribble


def test_generator_shape_and_determinism():
    tc,terrain,_=make()
    assert terrain.shape==(80,80)
    assert np.allclose(terrain,generate_terrain(tc))


def test_terrain_round_trip(tmp_path):
    tc,terrain,_=make(); path=save_terrain(tmp_path/'demo.terrain',tc,terrain); tc2,h2=load_terrain(path)
    assert tc2==tc; assert np.allclose(h2,terrain)


def test_fixed_clearance():
    tc,terrain,s=make(); cfg=TrajectoryConfig(path_resolution_m=.4,altitude_mode=AltitudeMode.TERRAIN_CLEARANCE,clearance_m=3.0)
    tr=build_trajectory(s,terrain,tc,cfg)
    assert np.allclose(tr.xyz[:,2]-tr.terrain_z,3.0)


def test_cruise_clears_and_limits_slope():
    tc,terrain,s=make(); cfg=TrajectoryConfig(path_resolution_m=.25,altitude_mode=AltitudeMode.CRUISE_ALTITUDE,clearance_m=2.0,cruise_altitude_m=4.0,max_climb_angle_deg=15)
    tr=build_trajectory(s,terrain,tc,cfg); required=np.maximum(cfg.cruise_altitude_m,tr.terrain_z+cfg.clearance_m)
    assert np.all(tr.xyz[:,2]>=required-1e-9)
    ds=np.linalg.norm(np.diff(tr.xy,axis=0),axis=1); dz=np.abs(np.diff(tr.xyz[:,2]))
    assert np.all(dz/np.maximum(ds,1e-12)<=np.tan(np.deg2rad(15))+1e-8)


def test_constant_3d_speed_and_covariance_sampling():
    tc,terrain,s=make(); tr=build_trajectory(s,terrain,tc,TrajectoryConfig(path_resolution_m=.25)); motion=MotionConfig(speed_mps=5,sample_dt_sec=.1,covariance_sample_rate_sec=.5,covariance_window_sec=2)
    samples=sample_motion(tr,terrain,tc,motion)
    assert np.allclose(np.linalg.norm(samples.velocity,axis=1),5)
    cov=sample_velocity_covariance(samples,motion)
    assert cov.matrices.shape[1:]==(3,3)
    assert len(cov.time_sec)==len(cov.matrices)
    assert np.all(np.linalg.eigvalsh(cov.matrices)>=-1e-9)


def test_targets_own_independent_motion_and_altitude_configs():
    from trajectory_app.models import TargetState

    first = TargetState("First", "#e74c3c")
    second = TargetState("Second", "#3498db")

    first.motion_config.speed_mps = 17.0
    first.trajectory_config.altitude_mode = AltitudeMode.CRUISE_ALTITUDE
    first.trajectory_config.cruise_altitude_m = 123.0

    assert second.motion_config.speed_mps != first.motion_config.speed_mps
    assert second.trajectory_config.altitude_mode == AltitudeMode.TERRAIN_CLEARANCE
    assert second.trajectory_config.cruise_altitude_m != first.trajectory_config.cruise_altitude_m


def test_solver_zero_covariance_returns_current_position():
    from trajectory_app.solver import PrincipalSigmaDemoSolver, SolverRequest

    request = SolverRequest(
        target_id="t",
        target_name="T",
        sample_index=3,
        time_sec=1.0,
        current_position=np.array([10.0, 20.0, 30.0]),
        current_velocity=np.array([4.0, 0.0, 0.0]),
        velocity_covariance=np.zeros((3, 3)),
        terrain_height_m=2.0,
        motion_config=MotionConfig(speed_mps=4.0, sample_dt_sec=0.25, covariance_sample_rate_sec=0.5),
        trajectory_config=TrajectoryConfig(),
    )
    result = PrincipalSigmaDemoSolver().solve(request)
    horizon = request.motion_config.prediction_dt_sec
    assert np.allclose(
        result.estimated_position,
        request.current_position + request.current_velocity * horizon,
    )


def test_solver_principal_sigma_offset_has_correct_units_and_axis():
    from trajectory_app.solver import PrincipalSigmaDemoSolver, SolverRequest

    request = SolverRequest(
        target_id="t",
        target_name="T",
        sample_index=3,
        time_sec=1.0,
        current_position=np.array([10.0, 20.0, 30.0]),
        current_velocity=np.array([4.0, 0.0, 0.0]),
        velocity_covariance=np.diag([4.0, 1.0, 0.25]),
        terrain_height_m=2.0,
        motion_config=MotionConfig(speed_mps=4.0, sample_dt_sec=0.25, covariance_sample_rate_sec=0.5),
        trajectory_config=TrajectoryConfig(),
    )
    # Default prediction_dt_sec is 2.0 s, so the nominal future center is
    # [18, 20, 30]. sigma_x = 2 m/s, giving a +4 m offset on X.
    result = PrincipalSigmaDemoSolver().solve(request)
    assert np.allclose(result.estimated_position, [22.0, 20.0, 30.0])
    assert np.isclose(result.metadata["offset_m"], 4.0)


def test_solver_statistics_mse_rmse_and_no_double_count():
    from trajectory_app.models import SolverStatistics

    stats = SolverStatistics.empty()

    stats.put(
        sample_index=0,
        time_sec=0.0,
        actual_position=np.array([0.0, 0.0, 0.0]),
        estimated_position=np.array([3.0, 4.0, 0.0]),
    )
    stats.put(
        sample_index=1,
        time_sec=1.0,
        actual_position=np.array([0.0, 0.0, 0.0]),
        estimated_position=np.array([0.0, 0.0, 0.0]),
    )

    assert stats.count == 2
    assert np.isclose(stats.mse_m2, 12.5)
    assert np.isclose(stats.rmse_m, np.sqrt(12.5))

    # Updating/repainting the same sample replaces it rather than double counting.
    stats.put(
        sample_index=0,
        time_sec=0.0,
        actual_position=np.array([0.0, 0.0, 0.0]),
        estimated_position=np.array([0.0, 0.0, 0.0]),
    )

    assert stats.count == 2
    assert np.isclose(stats.mse_m2, 0.0)
    assert np.isclose(stats.rmse_m, 0.0)


def test_terrain_surface_position_matches_heightmap():
    from trajectory_app.core import bilinear_height, terrain_surface_position

    terrain_config = TerrainConfig(
        x_min=-10.0,
        x_max=10.0,
        y_min=-5.0,
        y_max=5.0,
        grid_size=40,
        seed=13,
    )
    terrain = generate_terrain(terrain_config)
    position = terrain_surface_position(terrain, terrain_config, 1.25, -2.5)
    expected_z = float(
        bilinear_height(terrain, position[0], position[1], terrain_config)
    )
    assert np.isclose(position[2], expected_z)


def test_cameraman_state_owns_independent_view_sector():
    from trajectory_app.models import CameramanState

    a = CameramanState(
        "A",
        "#ff0000",
        view_radius_m=12.0,
        view_start_angle_deg=-20.0,
        view_end_angle_deg=35.0,
    )
    b = CameramanState(
        "B",
        "#00ff00",
        view_radius_m=5.0,
        view_start_angle_deg=90.0,
        view_end_angle_deg=180.0,
    )
    assert a.view_radius_m == 12.0
    assert b.view_radius_m == 5.0
    assert (a.view_start_angle_deg, a.view_end_angle_deg) != (
        b.view_start_angle_deg,
        b.view_end_angle_deg,
    )


def test_view_sector_wraparound_geometry():
    from trajectory_app.core import view_sector_boundary_xy, view_sector_span_deg

    assert np.isclose(view_sector_span_deg(330.0, 30.0), 60.0)
    boundary = view_sector_boundary_xy(0.0, 0.0, 10.0, 330.0, 30.0)
    assert len(boundary) > 2
    assert np.allclose(np.linalg.norm(boundary, axis=1), 10.0)
    assert np.allclose(boundary[0], [10.0 * np.cos(np.deg2rad(330.0)), 10.0 * np.sin(np.deg2rad(330.0))])
    assert np.allclose(boundary[-1], [10.0 * np.cos(np.deg2rad(390.0)), 10.0 * np.sin(np.deg2rad(390.0))])


def test_prediction_ellipse_propagates_velocity_covariance():
    from trajectory_app.core import predict_position_ellipse_xy

    position = np.array([10.0, 20.0, 30.0])
    velocity = np.array([3.0, 4.0, 5.0])
    covariance = np.diag([4.0, 1.0, 9.0])

    center, ellipse = predict_position_ellipse_xy(
        position,
        velocity,
        covariance,
        2.0,
        point_count=64,
    )

    assert np.allclose(center, [16.0, 28.0])
    assert ellipse.shape == (64, 2)

    # Σp = 4 * Σv, so XY 1σ radii are 4 m and 2 m.
    delta = ellipse - center
    assert np.isclose(np.max(np.abs(delta[:, 0])), 4.0)
    assert np.isclose(np.max(np.abs(delta[:, 1])), 2.0)


def test_prediction_ellipse_zero_covariance_collapses_to_center():
    from trajectory_app.core import predict_position_ellipse_xy

    center, ellipse = predict_position_ellipse_xy(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, -1.0, 0.0]),
        np.zeros((3, 3)),
        2.0,
    )

    assert np.allclose(center, [5.0, 0.0])
    assert np.allclose(ellipse, center)


def test_prediction_horizon_defaults_to_two_seconds():
    from trajectory_app.models import MotionConfig

    assert MotionConfig().prediction_dt_sec == 2.0
