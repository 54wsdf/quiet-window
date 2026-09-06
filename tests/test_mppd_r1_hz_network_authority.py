from scripts.mppd_r1_hz_network_authority import build_authority


def route_support_fixture():
    return {
        "status": "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT",
        "route_support": {
            "A:67->B:5": [
                {
                    "ride_legs": [
                        {
                            "from_station": 67,
                            "to_station": 5,
                            "compatible_service_options": [{"path_id": "A_main", "direction": "Down"}],
                        },
                        {
                            "from_station": 5,
                            "to_station": 4,
                            "compatible_service_options": [{"path_id": "B_main", "direction": "Up"}],
                        },
                    ]
                }
            ],
            "B:4->B:28": [
                {
                    "ride_legs": [
                        {
                            "from_station": 4,
                            "to_station": 20,
                            "compatible_service_options": [{"path_id": "B_main", "direction": "Down"}],
                        },
                        {
                            "from_station": 20,
                            "to_station": 28,
                            "compatible_service_options": [{"path_id": "B_branch", "direction": "Down"}],
                        },
                    ]
                }
            ],
        },
    }


def test_authority_contains_network_but_no_service_count_or_event_inventory():
    a = build_authority(route_support_fixture())
    assert a["schema"] == "mppd.r1-network-authority.v2"
    assert len(a["station_ids"]) == 81
    assert len(a["line_paths"]) == 8
    assert "service_event_keys" not in a
    assert "expected_service_count" not in a
    assert a["semantics"]["service_count_must_be_inferred"] is True


def test_transfer_movements_are_derived_from_route_topology_and_direction():
    a = build_authority(route_support_fixture())
    keys = {
        (
            x["station_id"],
            x["from_line_id"],
            x["from_direction_id"],
            x["to_line_id"],
            x["to_direction_id"],
        )
        for x in a["transfer_movements"]
    }
    assert ("5", "A", "Down", "B", "Up") in keys
    assert ("20", "B", "Down", "B", "Down") in keys
