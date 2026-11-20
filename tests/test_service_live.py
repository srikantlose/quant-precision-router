import pytest


@pytest.mark.gpu
def test_live_service_check(tmp_path):
    from quantlab.service_check import run_service_check

    result = run_service_check(out_path=tmp_path / "service_check.json", log=lambda m: None)
    assert result["passed"], result["routing_checks"]
    assert len(result["determinism"]) == 5
