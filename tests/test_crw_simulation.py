import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
import numpy as np

from .crw_test_utils import (
    ThreeChoiceTurnFlightParameters,
    build_argument_parser,
    compute_endpoint_statistics,
    print_three_choice_turn_flight_report,
    simulate_three_choice_turn_flight_endpoints,
)


class ThreeChoiceTurnFlightEndpointTests(unittest.TestCase):
    def test_parser_defaults_to_crw_and_upward_initial_direction(self) -> None:
        arguments = build_argument_parser().parse_args([])

        self.assertEqual(arguments.model, "crw")
        self.assertEqual(arguments.initial_direction, np.pi / 2)
        self.assertIsNone(arguments.time_step)

    def test_parser_accepts_three_choice_options(self) -> None:
        arguments = build_argument_parser().parse_args([
            "--model", "three-choice-turn-flight", "--speed", "2.0",
            "--turning-radius", "5.0", "--time-step", "0.1",
            "--initial-direction", "0.25",
        ])

        self.assertEqual(arguments.model, "three-choice-turn-flight")
        self.assertEqual(arguments.speed, 2.0)
        self.assertEqual(arguments.turning_radius, 5.0)
        self.assertEqual(arguments.time_step, 0.1)
        self.assertEqual(arguments.initial_direction, 0.25)

    def test_endpoints_are_batched_shaped_and_reproducible(self) -> None:
        parameters = ThreeChoiceTurnFlightParameters(
            n_steps=4,
            speed=2.0,
            turning_radius=5.0,
            time_step=0.1,
            initial_direction=np.pi / 2,
            n_trajectories=7,
            seed=12,
            batch_size=3,
        )

        first = simulate_three_choice_turn_flight_endpoints(parameters)
        second = simulate_three_choice_turn_flight_endpoints(parameters)

        self.assertEqual(first.shape, (7, 2))
        np.testing.assert_array_equal(first, second)

    def test_rejects_missing_or_non_positive_physical_parameters(self) -> None:
        for parameter, value in (("speed", 0.0), ("turning_radius", 0.0), ("time_step", None), ("time_step", 0.0)):
            values = {
                "n_steps": 1,
                "speed": 1.0,
                "turning_radius": 1.0,
                "time_step": 1.0,
                "initial_direction": 0.0,
                "n_trajectories": 1,
                "seed": 1,
                "batch_size": 1,
            }
            values[parameter] = value
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                simulate_three_choice_turn_flight_endpoints(
                    ThreeChoiceTurnFlightParameters(**values),
                )

    def test_report_contains_empirical_parameters_without_crw_theory(self) -> None:
        parameters = ThreeChoiceTurnFlightParameters(
            n_steps=2,
            speed=1.0,
            turning_radius=2.0,
            time_step=0.1,
            initial_direction=np.pi / 2,
            n_trajectories=3,
            seed=1,
            batch_size=3,
        )
        statistics = compute_endpoint_statistics(np.array([[0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))
        output = StringIO()

        with redirect_stdout(output):
            print_three_choice_turn_flight_report(parameters, statistics)

        self.assertIn("THREE-CHOICE TURN FLIGHT", output.getvalue())
        self.assertIn("speed=", output.getvalue())
        self.assertIn("turning_radius=", output.getvalue())
        self.assertNotIn("Exact MSD", output.getvalue())
