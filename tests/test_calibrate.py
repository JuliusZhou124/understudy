import numpy as np

from understudy.calibrate import coverage, crps_sample


def test_coverage_counts_actuals_inside_the_interval():
    preds = [np.array([100.0, 200.0, 300.0])] * 4
    actuals = [150.0, 250.0, 50.0, 500.0]
    assert coverage(preds, actuals, lo=0.1, hi=0.9) == 0.5


def test_coverage_is_one_when_everything_falls_inside():
    preds = [np.array([0.0, 1000.0])] * 3
    assert coverage(preds, [100.0, 200.0, 300.0]) == 1.0


def test_crps_is_zero_for_a_perfect_point_forecast():
    assert crps_sample(np.array([100.0] * 50), 100.0) == 0.0


def test_crps_grows_with_error():
    s = np.array([100.0] * 50)
    assert crps_sample(s, 150.0) > crps_sample(s, 110.0)


def test_crps_rewards_a_sharp_correct_forecast_over_a_diffuse_one():
    sharp = np.array([100.0] * 50)
    diffuse = np.linspace(0.0, 200.0, 50)
    assert crps_sample(sharp, 100.0) < crps_sample(diffuse, 100.0)
