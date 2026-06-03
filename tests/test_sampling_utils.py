import numpy as np

from Upsampling.sampling_utils import clamp_sample_count, safe_index_array


def test_clamp_sample_count_keeps_request_inside_available_range():
    assert clamp_sample_count(10, 4, minimum=1) == 4
    assert clamp_sample_count(0, 4, minimum=1) == 1
    assert clamp_sample_count(3, 4, minimum=1) == 3


def test_safe_index_array_drops_out_of_bounds_indices_and_fills_from_available_points():
    idx = safe_index_array([0, 4, -1, 2], available_count=4, desired_count=3)

    assert idx.tolist() == [0, 2, 1]


def test_safe_index_array_handles_empty_or_invalid_sampler_output():
    idx = safe_index_array([9, -3], available_count=3, desired_count=2)

    assert idx.tolist() == [0, 1]
