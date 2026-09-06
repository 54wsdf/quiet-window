from collections import Counter

from scripts.mppd_r1_hz_directional_transfer_evidence import (
    relation_candidates,
    weighted_quantile,
)


def service(path_id, direction, options):
    return {
        "path_id": path_id,
        "direction": direction,
        "options": set(options),
    }


def leg(options):
    return {
        "compatible_service_options": [
            {"path_id": p, "direction": d} for p, d in options
        ]
    }


def test_weighted_quantiles_preserve_passenger_mass():
    h = Counter({10.0: 1.0, 20.0: 2.0, 30.0: 1.0})
    assert weighted_quantile(h, 0.05) == 10.0
    assert weighted_quantile(h, 0.50) == 20.0
    assert weighted_quantile(h, 0.95) == 30.0


def test_directional_relation_candidates_keep_service_option_ambiguity():
    option_line = {
        ("A_main", "Down"): "A",
        ("B_main", "Up"): "B",
        ("B_branch", "Up"): "B",
    }
    authority_moves = {
        ("5", "A", "Down", "B", "Up"): {"movement_id": "m1"},
    }
    left = service("A_main", "Down", {("A_main", "Down")})
    right = service("B_main", "Up", {("B_main", "Up"), ("B_branch", "Up")})
    candidates = relation_candidates(
        5,
        left,
        right,
        leg({("A_main", "Down")}),
        leg({("B_main", "Up"), ("B_branch", "Up")}),
        option_line,
        authority_moves,
    )
    # Both downstream path hypotheses map to the same physical line/direction
    # transfer relation, so ambiguity is retained without double-counting it.
    assert candidates == {("5", "A", "Down", "B", "Up")}


def test_candidate_not_in_network_authority_is_not_invented():
    option_line = {
        ("A_main", "Down"): "A",
        ("C_main", "Up"): "C",
    }
    candidates = relation_candidates(
        5,
        service("A_main", "Down", {("A_main", "Down")}),
        service("C_main", "Up", {("C_main", "Up")}),
        leg({("A_main", "Down")}),
        leg({("C_main", "Up")}),
        option_line,
        {},
    )
    assert candidates == set()
