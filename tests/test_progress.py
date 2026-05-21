from wind_forecast.progress import _fmt_duration


def test_fmt_duration_units() -> None:
    assert _fmt_duration(-3) == "0.0s"
    assert _fmt_duration(0) == "0.0s"
    assert _fmt_duration(0.5) == "0.5s"
    assert _fmt_duration(59.4) == "59.4s"
    assert _fmt_duration(60) == "1m00s"
    assert _fmt_duration(90) == "1m30s"
    assert _fmt_duration(3599) == "59m59s"
    assert _fmt_duration(3600) == "1h00m"
    assert _fmt_duration(3700) == "1h01m"
    assert _fmt_duration(86399) == "23h59m"
    assert _fmt_duration(86400) == "1d00h"
    assert _fmt_duration(200_000) == "2d07h"
