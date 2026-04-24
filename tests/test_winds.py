"""Wind vector conversion tests."""

from __future__ import annotations

import numpy as np

from wind_forecast.winds import dir_speed_to_uv, uv_to_dir_speed


def test_north_wind_has_negative_v() -> None:
    # Wind FROM 360/0 at 10kt: the air is moving south -> v is negative.
    u, v = dir_speed_to_uv(np.array([0.0]), np.array([10.0]))
    assert np.isclose(u[0], 0.0, atol=1e-9)
    assert np.isclose(v[0], -10.0)


def test_east_wind_has_negative_u() -> None:
    # Wind FROM 90 at 10kt: air moves west -> u is negative.
    u, v = dir_speed_to_uv(np.array([90.0]), np.array([10.0]))
    assert np.isclose(u[0], -10.0)
    assert np.isclose(v[0], 0.0, atol=1e-9)


def test_roundtrip_preserves_direction_and_speed() -> None:
    directions = np.array([10.0, 45.0, 90.0, 180.0, 270.0, 359.0])
    speeds = np.array([3.0, 7.5, 12.0, 20.0, 5.0, 30.0])
    u, v = dir_speed_to_uv(directions, speeds)
    d2, s2 = uv_to_dir_speed(u, v)
    assert np.allclose(s2, speeds)
    # Direction wraps — compare mod 360
    diff = (d2 - directions + 540) % 360 - 180
    assert np.allclose(diff, 0.0, atol=1e-6)


def test_calm_wind_stays_zero() -> None:
    u, v = dir_speed_to_uv(np.array([0.0]), np.array([0.0]))
    assert u[0] == 0.0 and v[0] == 0.0
