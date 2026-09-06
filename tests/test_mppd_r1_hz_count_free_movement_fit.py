import math

import scripts.mppd_r1_hz_count_free_movement_fit as m


def test_access_boarding_integrates_waiting_instead_of_equating_access_to_gap():
    r = m.boarding_moments(
        lower=15.0,
        base_time=100.0,
        selected_time=180.0,
        previous_times=[140.0],
        mu=math.log(35.0),
        sigma=0.45,
        hazard=0.8,
    )
    assert r is not None
    assert 15.0 <= math.exp(r["elog"]) < 80.0
    assert r["eskip"] >= 0.0


def test_transfer_lower_bound_is_five_seconds():
    assert m.boarding_moments(5.0, 100.0, 104.0, [], math.log(30.0), 0.5, 0.8) is None
    r = m.boarding_moments(5.0, 100.0, 110.0, [], math.log(30.0), 0.5, 0.8)
    assert r is not None
    assert math.exp(r["elog"]) >= 5.0


def test_truncated_fit_respects_positive_scale_and_bound():
    xs = [25.0, 30.0, 40.0, 55.0, 70.0]
    w = float(len(xs))
    ys = [math.log(x) for x in xs]
    mu, sigma = m.fit_truncated_from_expected(15.0, w, sum(ys), sum(y*y for y in ys), math.log(45.0), 0.6)
    assert math.isfinite(mu)
    assert 0.15 <= sigma <= 1.8
    assert m.truncated_quantile(15.0, mu, sigma, 0.05) >= 15.0


def test_transfer_default_has_multiple_latent_paths():
    x = m.transfer_default()
    assert len(x["components"]) >= 2
    assert {c["path_id"] for c in x["components"]} == {"latent_path_1", "latent_path_2"}
