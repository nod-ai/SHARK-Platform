# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pytest
import json
from subprocess import CompletedProcess
from unittest.mock import call, patch, MagicMock
from . import libtuner

"""
Usage: python -m pytest libtuner_test.py
"""


def test_group_benchmark_results_by_device_id() -> None:
    # Create mock TaskResult objects with device_id attributes
    task_result_1: libtuner.TaskResult = MagicMock(spec=libtuner.TaskResult)
    task_result_1.device_id = "device_1"

    task_result_2: libtuner.TaskResult = MagicMock(spec=libtuner.TaskResult)
    task_result_2.device_id = "device_2"

    task_result_3: libtuner.TaskResult = MagicMock(spec=libtuner.TaskResult)
    task_result_3.device_id = "device_1"

    benchmark_results = [task_result_1, task_result_2, task_result_3]

    expected_grouped_results = [
        [task_result_1, task_result_3],  # Grouped by device_1
        [task_result_2],  # Grouped by device_2
    ]

    grouped_results = libtuner.group_benchmark_results_by_device_id(benchmark_results)

    assert grouped_results == expected_grouped_results
    assert grouped_results[0][0].device_id == "device_1"
    assert grouped_results[1][0].device_id == "device_2"


def test_find_collisions() -> None:
    input = [(1, "abc"), (2, "def"), (3, "abc")]
    assert libtuner.find_collisions(input) == (True, [("abc", [1, 3]), ("def", [2])])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert libtuner.find_collisions(input) == (
        False,
        [("abc", [1]), ("def", [2]), ("hig", [3])],
    )


def test_collision_handler() -> None:
    input = [(1, "abc"), (2, "def"), (3, "abc"), (4, "def"), (5, "hig")]
    assert libtuner.collision_handler(input) == (True, [1, 2, 5])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert libtuner.collision_handler(input) == (False, [])


def test_IREEBenchmarkResult_get() -> None:
    # Time is int in us
    int_json = [{"aggregate_name": "mean", "real_time": 1, "time_unit": "us"}]

    res = libtuner.IREEBenchmarkResult(candidate_id=1, result_json=int_json)
    assert res.get_mean_time_us() == float(1)

    # Time is float in us
    float_json = [{"aggregate_name": "mean", "real_time": 123.45, "time_unit": "us"}]

    res = libtuner.IREEBenchmarkResult(candidate_id=2, result_json=float_json)
    assert res.get_mean_time_us() == 123.45

    # Time is in seconds
    seconds_json = [{"aggregate_name": "mean", "real_time": 1.0, "time_unit": "s"}]

    res = libtuner.IREEBenchmarkResult(candidate_id=3, result_json=seconds_json)
    assert res.get_mean_time_us() == 1.0 * 1e6

    # Time is in miliseconds
    miliseconds_json = [{"aggregate_name": "mean", "real_time": 1.0, "time_unit": "ms"}]

    res = libtuner.IREEBenchmarkResult(candidate_id=4, result_json=miliseconds_json)
    assert res.get_mean_time_us() == 1.0 * 1e3

    # Time is in nanoseconds
    nanoseconds_json = [{"aggregate_name": "mean", "real_time": 1.0, "time_unit": "ns"}]

    res = libtuner.IREEBenchmarkResult(candidate_id=5, result_json=nanoseconds_json)
    assert res.get_mean_time_us() == 1.0 * 1e-3

    small_number_json = [
        {
            "aggregate_name": "mean",
            "real_time": 3.4591828516259519e-02,
            "time_unit": "ms",
        }
    ]

    res = libtuner.IREEBenchmarkResult(candidate_id=6, result_json=small_number_json)
    assert res.get_mean_time_us() == 34.591828516259519

    # Invalid json: missing real_time
    invalid_real_time_json = [{"aggregate_name": "mean", "real_time": None}]

    res = libtuner.IREEBenchmarkResult(
        candidate_id=7, result_json=invalid_real_time_json
    )
    assert res.get_mean_time_us() == None

    # Invalid json: empty dictionary
    res = libtuner.IREEBenchmarkResult(candidate_id=8, result_json=[])
    assert res.get_mean_time_us() is None

    # Invalid json: invalid time unit
    invalid_time_unit_json = [
        {"aggregate_name": "mean", "real_time": 1.0, "time_unit": "invalid_unit"}
    ]

    with pytest.raises(AssertionError, match="Unsupported time unit: invalid_unit"):
        res = libtuner.IREEBenchmarkResult(
            candidate_id=9, result_json=invalid_time_unit_json
        )
        res.get_mean_time_us()

    # Invalid json: missing aggregate_name
    invalid_aggregate_name_json = [{"real_time": 1.0, "time_unit": "us"}]

    res = libtuner.IREEBenchmarkResult(
        candidate_id=10, result_json=invalid_aggregate_name_json
    )
    assert res.get_mean_time_us() is None


def test_generate_display_BR() -> None:
    output = libtuner.generate_display_DBR(1, 3.14)
    expected = f"1\tMean Time: 3.1"
    assert output == expected, "DispatchBenchmarkResult generates invalid sample string"

    output = libtuner.generate_display_MBR("baseline.vmfb", str(1), 567.89)
    expected = "Benchmarking: baseline.vmfb on device 1: 568"
    assert output == expected, "ModelBenchmarkResult generates invalid sample string"
    output = libtuner.generate_display_MBR("baseline.vmfb", str(1), 567.89, 0.0314)
    expected = "Benchmarking: baseline.vmfb on device 1: 568 (+3.140%)"
    assert output == expected, "ModelBenchmarkResult generates invalid sample string"
    output = libtuner.generate_display_MBR("baseline.vmfb", str(1), 567.89, -3.14)
    expected = "Benchmarking: baseline.vmfb on device 1: 568 (-314.000%)"
    assert output == expected, "ModelBenchmarkResult generates invalid sample string"


def make_mock_task_result() -> libtuner.TaskResult:
    process: CompletedProcess = MagicMock(spec=CompletedProcess)
    run_result = libtuner.RunResult(process, False)
    task_result = libtuner.TaskResult(run_result, 0, "")
    return task_result


def test_parse_dispatch_benchmark_results() -> None:
    base_path = libtuner.Path("/mock/base/dir")
    spec_dir = base_path / "specs"
    path_config = libtuner.PathConfig()
    object.__setattr__(path_config, "specs_dir", spec_dir)

    mock_result_1 = make_mock_task_result()
    mock_json_1 = {
        "benchmarks": [
            {"aggregate_name": "mean", "real_time": 100.0, "time_unit": "us"}
        ]
    }
    assert mock_result_1.run_result.process_res is not None
    mock_result_1.run_result.process_res.stdout = json.dumps(mock_json_1)
    mock_result_1.candidate_id = 1
    mock_result_2 = make_mock_task_result()
    mock_json_2 = {
        "benchmarks": [
            {"aggregate_name": "mean", "real_time": 200.0, "time_unit": "us"}
        ]
    }
    assert mock_result_2.run_result.process_res is not None
    mock_result_2.run_result.process_res.stdout = json.dumps(mock_json_2)
    mock_result_2.candidate_id = 2
    mock_result_3 = make_mock_task_result()
    mock_json_3 = {
        "benchmarks": [
            {
                "aggregate_name": "mean",
                "real_time": 3.4591828516259519e-02,
                "time_unit": "ms",
            }
        ]
    }
    assert mock_result_3.run_result.process_res is not None
    mock_result_3.run_result.process_res.stdout = json.dumps(mock_json_3)
    mock_result_3.candidate_id = 3
    # Incomplete result.
    mock_result_4 = libtuner.TaskResult(libtuner.RunResult(None, True), 4, "4")
    benchmark_results = [mock_result_1, mock_result_2, mock_result_3, mock_result_4]

    candidate_trackers = []
    for i in range(4):
        tracker = libtuner.CandidateTracker(candidate_id=i)
        tracker.dispatch_mlir_path = libtuner.Path(f"/mock/mlir/path/{i}.mlir")
        candidate_trackers.append(tracker)

    expected_parsed_results = [
        libtuner.ParsedDisptachBenchmarkResult(
            candidate_id=1,
            benchmark_time_in_seconds=100.0,
            candidate_mlir=libtuner.Path("/mock/mlir/path/1.mlir"),
            candidate_spec_mlir=libtuner.Path("/mock/base/dir/specs/1_spec.mlir"),
        ),
        libtuner.ParsedDisptachBenchmarkResult(
            candidate_id=2,
            benchmark_time_in_seconds=200.0,
            candidate_mlir=libtuner.Path("/mock/mlir/path/2.mlir"),
            candidate_spec_mlir=libtuner.Path("/mock/base/dir/specs/2_spec.mlir"),
        ),
        libtuner.ParsedDisptachBenchmarkResult(
            candidate_id=3,
            benchmark_time_in_seconds=34.591828516259519,
            candidate_mlir=libtuner.Path("/mock/mlir/path/3.mlir"),
            candidate_spec_mlir=libtuner.Path("/mock/base/dir/specs/3_spec.mlir"),
        ),
    ]
    expected_dump_list = [
        "1\tMean Time: 100.0\n",
        "2\tMean Time: 200.0\n",
        "3\tMean Time: 34.6\n",
        "Candidate 4 not completed",
    ]

    parsed_results, dump_list = libtuner.parse_dispatch_benchmark_results(
        path_config, benchmark_results, candidate_trackers
    )

    assert parsed_results == expected_parsed_results
    assert dump_list == expected_dump_list
    assert candidate_trackers[1].first_benchmark_time == 100.0
    assert candidate_trackers[1].spec_path == libtuner.Path(
        "/mock/base/dir/specs/1_spec.mlir"
    )
    assert candidate_trackers[2].first_benchmark_time == 200.0
    assert candidate_trackers[2].spec_path == libtuner.Path(
        "/mock/base/dir/specs/2_spec.mlir"
    )
    assert candidate_trackers[3].first_benchmark_time == 34.591828516259519
    assert candidate_trackers[3].spec_path == libtuner.Path(
        "/mock/base/dir/specs/3_spec.mlir"
    )


def test_parse_model_benchmark_results() -> None:
    # Setup mock data for candidate_trackers
    tracker0 = libtuner.CandidateTracker(0)
    tracker0.compiled_model_path = libtuner.Path("/path/to/baseline.vmfb")

    tracker1 = libtuner.CandidateTracker(1)
    tracker1.compiled_model_path = libtuner.Path("/path/to/model_1.vmfb")

    tracker2 = libtuner.CandidateTracker(2)
    tracker2.compiled_model_path = libtuner.Path("/path/to/model_2.vmfb")

    tracker3 = libtuner.CandidateTracker(3)
    tracker3.compiled_model_path = libtuner.Path("/path/to/model_3.vmfb")

    candidate_trackers = [tracker0, tracker1, tracker2, tracker3]

    # Setup mock data for task results
    result1 = make_mock_task_result()
    result_json_1 = {"benchmarks": [{"real_time": 1.23}]}
    assert result1.run_result.process_res is not None
    result1.run_result.process_res.stdout = json.dumps(result_json_1)
    result1.candidate_id = 1
    result1.device_id = "device1"

    result2 = make_mock_task_result()
    result_json_2 = {"benchmarks": [{"real_time": 4.56}]}
    assert result2.run_result.process_res is not None
    result2.run_result.process_res.stdout = json.dumps(result_json_2)
    result2.candidate_id = 2
    result2.device_id = "device2"

    result3 = make_mock_task_result()
    result_json_3 = {"benchmarks": [{"real_time": 0.98}]}
    assert result3.run_result.process_res is not None
    result3.run_result.process_res.stdout = json.dumps(result_json_3)
    result3.candidate_id = 0
    result3.device_id = "device1"

    result4 = make_mock_task_result()
    result_json_4 = {"benchmarks": [{"real_time": 4.13}]}
    assert result4.run_result.process_res is not None
    result4.run_result.process_res.stdout = json.dumps(result_json_4)
    result4.candidate_id = 0
    result4.device_id = "device2"

    # Incomplete baseline on device3
    result5 = libtuner.TaskResult(libtuner.RunResult(None, True), 0, "device3")

    result6 = make_mock_task_result()
    result_json_6 = {"benchmarks": [{"real_time": 3.38}]}
    assert result6.run_result.process_res is not None
    result6.run_result.process_res.stdout = json.dumps(result_json_6)
    result6.candidate_id = 3
    result6.device_id = "device3"

    candidate_results = [result1, result2, result6]
    baseline_results = [result3, result4, result5]

    # Skip real benchmark extraction, directly use given values from above
    def mock_get_mean_time_us(self):
        return float(self.result_json[0]["real_time"]) if self.result_json else None

    # Mock IREEBenchmarkResult to return wanted benchmark times
    with patch(
        f"{libtuner.__name__}.IREEBenchmarkResult.get_mean_time_us",
        new=mock_get_mean_time_us,
    ):
        # Mock handle_error to avoid actual logging during tests
        with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
            dump_list = libtuner.parse_model_benchmark_results(
                candidate_trackers, candidate_results, baseline_results
            )

            # Verify interactions with candidate_trackers
            assert tracker1.model_benchmark_time == 1.23
            assert tracker1.model_benchmark_device_id == "device1"
            assert tracker1.baseline_benchmark_time == 0.98
            assert tracker1.calibrated_benchmark_diff == pytest.approx(
                (1.23 - 0.98) / 0.98, rel=1e-6
            )

            assert tracker2.model_benchmark_time == 4.56
            assert tracker2.model_benchmark_device_id == "device2"
            assert tracker2.baseline_benchmark_time == 4.13
            assert tracker2.calibrated_benchmark_diff == pytest.approx(
                (4.56 - 4.13) / 4.13, rel=1e-6
            )

            assert tracker3.model_benchmark_time == 3.38
            assert tracker3.model_benchmark_device_id == "device3"

            assert dump_list == [
                "Benchmarking: /path/to/baseline.vmfb on device device1: 0.98\n" "\n",
                "Benchmarking: /path/to/model_1.vmfb on device device1: 1.23 (+25.510%)\n"
                "\n",
                "Benchmarking: /path/to/baseline.vmfb on device device2: 4.13\n" "\n",
                "Benchmarking: /path/to/model_2.vmfb on device device2: 4.56 (+10.412%)\n"
                "\n",
                "Benchmarking: /path/to/model_3.vmfb on device device3: 3.38\n" "\n",
                "Benchmarking result of /path/to/baseline.vmfb on device device3 is incomplete\n",
            ]

            # Verify handle_error was called correctly
            mock_handle_error.assert_called_once_with(
                condition=True,
                msg="Benchmarking result of /path/to/baseline.vmfb on device device3 is incomplete",
                level=libtuner.logging.WARNING,
            )


def test_extract_driver_names() -> None:
    user_devices = ["hip://0", "local-sync://default", "cuda://default"]
    expected_output = {"hip", "local-sync", "cuda"}

    assert libtuner.extract_driver_names(user_devices) == expected_output


def test_fetch_available_devices_success() -> None:
    drivers = ["hip", "local-sync", "cuda"]
    mock_devices = {
        "hip": [{"path": "ABCD", "device_id": 1}],
        "local-sync": [{"path": "default", "device_id": 2}],
        "cuda": [{"path": "default", "device_id": 3}],
    }

    with patch(f"{libtuner.__name__}.ireert.get_driver") as mock_get_driver:
        mock_driver = MagicMock()

        def get_mock_driver(name):
            mock_driver.query_available_devices.side_effect = lambda: mock_devices[name]
            return mock_driver

        mock_get_driver.side_effect = get_mock_driver

        actual_output = libtuner.fetch_available_devices(drivers)
        expected_output = [
            "hip://ABCD",
            "hip://0",
            "local-sync://default",
            "local-sync://1",
            "cuda://default",
            "cuda://2",
        ]

        assert actual_output == expected_output


def test_fetch_available_devices_failure() -> None:
    drivers = ["hip", "local-sync", "cuda"]
    mock_devices = {
        "hip": [{"path": "ABCD", "device_id": 1}],
        "local-sync": ValueError("Failed to initialize"),
        "cuda": [{"path": "default", "device_id": 1}],
    }

    with patch(f"{libtuner.__name__}.ireert.get_driver") as mock_get_driver:
        with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
            mock_driver = MagicMock()

            def get_mock_driver(name):
                if isinstance(mock_devices[name], list):
                    mock_driver.query_available_devices.side_effect = (
                        lambda: mock_devices[name]
                    )
                else:
                    mock_driver.query_available_devices.side_effect = lambda: (
                        _ for _ in ()
                    ).throw(mock_devices[name])
                return mock_driver

            mock_get_driver.side_effect = get_mock_driver

            actual_output = libtuner.fetch_available_devices(drivers)
            expected_output = ["hip://ABCD", "hip://0", "cuda://default", "cuda://0"]

            assert actual_output == expected_output
            mock_handle_error.assert_called_once_with(
                condition=True,
                msg="Could not initialize driver local-sync: Failed to initialize",
                error_type=ValueError,
                exit_program=True,
            )


def test_parse_devices() -> None:
    user_devices_str = "hip://0, local-sync://default, cuda://default"
    expected_output = ["hip://0", "local-sync://default", "cuda://default"]

    with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
        actual_output = libtuner.parse_devices(user_devices_str)
        assert actual_output == expected_output

        mock_handle_error.assert_not_called()


def test_parse_devices_with_invalid_input() -> None:
    user_devices_str = "hip://0, local-sync://default, invalid_device, cuda://default"
    expected_output = [
        "hip://0",
        "local-sync://default",
        "invalid_device",
        "cuda://default",
    ]

    with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
        actual_output = libtuner.parse_devices(user_devices_str)
        assert actual_output == expected_output

        mock_handle_error.assert_called_once_with(
            condition=True,
            msg=f"Invalid device list: {user_devices_str}. Error: {ValueError()}",
            error_type=argparse.ArgumentTypeError,
        )


def test_validate_devices() -> None:
    user_devices = ["hip://0", "local-sync://default"]
    user_drivers = {"hip", "local-sync"}

    with patch(f"{libtuner.__name__}.extract_driver_names", return_value=user_drivers):
        with patch(
            f"{libtuner.__name__}.fetch_available_devices",
            return_value=["hip://0", "local-sync://default"],
        ):
            with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
                libtuner.validate_devices(user_devices)
                assert all(
                    call[1]["condition"] is False
                    for call in mock_handle_error.call_args_list
                )


def test_validate_devices_with_invalid_device() -> None:
    user_devices = ["hip://0", "local-sync://default", "cuda://default"]
    user_drivers = {"hip", "local-sync", "cuda"}

    with patch(f"{libtuner.__name__}.extract_driver_names", return_value=user_drivers):
        with patch(
            f"{libtuner.__name__}.fetch_available_devices",
            return_value=["hip://0", "local-sync://default"],
        ):
            with patch(f"{libtuner.__name__}.handle_error") as mock_handle_error:
                libtuner.validate_devices(user_devices)
                expected_call = call(
                    condition=True,
                    msg=f"Invalid device specified: cuda://default\nFetched available devices: ['hip://0', 'local-sync://default']",
                    error_type=argparse.ArgumentError,
                    exit_program=True,
                )
                assert expected_call in mock_handle_error.call_args_list
