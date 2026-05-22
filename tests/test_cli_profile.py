from click.testing import CliRunner

from wind_forecast.cli import PROFILES, _apply_profile, cli


def test_apply_profile_no_profile_is_passthrough() -> None:
    out = _apply_profile(None, set(), lead_max=18, step_hours=1, hrrr_workers=8)
    assert out == {"lead_max": 18, "step_hours": 1, "hrrr_workers": 8}


def test_apply_profile_overrides_defaults() -> None:
    out = _apply_profile("pi", set(), lead_max=18, step_hours=1, hrrr_workers=8)
    assert out["lead_max"] == PROFILES["pi"]["lead_max"]
    assert out["step_hours"] == PROFILES["pi"]["step_hours"]
    assert out["hrrr_workers"] == PROFILES["pi"]["hrrr_workers"]


def test_apply_profile_explicit_flags_win() -> None:
    out = _apply_profile(
        "pi", {"lead_max", "step_hours"},
        lead_max=18, step_hours=1, hrrr_workers=8,
    )
    assert out["lead_max"] == 18
    assert out["step_hours"] == 1
    assert out["hrrr_workers"] == PROFILES["pi"]["hrrr_workers"]


def test_apply_profile_unknown_name_is_passthrough() -> None:
    out = _apply_profile("nonexistent", set(), lead_max=18)
    assert out == {"lead_max": 18}


def test_run_help_lists_profile_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "pi" in result.output
    assert "--step-hours" in result.output
