"""Regression tests for the required-precision labelling rule (plan Sec 9.2).
These exact cases came out of three rounds of adversarial review of the
methodology -- a silent regression here would quietly change what "required
precision" means without any other symptom."""

import pytest

from quantlab.labels import required_precision


@pytest.mark.parametrize("s,expected", [
    ([True, True, True, True, True, True, True], 0),
    ([False, True, True, True, True, True, True], 1),
    ([True, False, True, True, True, True, True], 0),   # one exception allowed
    ([True, False, False, True, True, True, True], 3),
    ([False, False, False, False, False, False, True], 6),
    ([False, False, False, False, False, False, False], None),
    ([False, False, False, False, False, True, False], 5),  # F16 not required
    ([False, False, False, False, True, False, False], 4),  # exception at Q8_0 allowed
])
def test_required_precision_cases(s, expected):
    assert required_precision(s) == expected


def test_max_exceptions_above_is_configurable():
    s = [True, False, False, False, True, True, True]  # 3 exceptions after Q2_K
    assert required_precision(s, max_exc=1) != 0
    assert required_precision(s, max_exc=3) == 0
