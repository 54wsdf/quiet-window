from scripts.mppd_r1_hz_operating_authority import (
    build_operating_authority,
    clock_to_service_seconds,
    normal_headway_floor_for,
    validate_operating_authority,
)


def test_operating_authority_is_count_free_and_depot_mapping_fails_closed():
    authority = build_operating_authority()
    assert validate_operating_authority(authority) == []
    semantics = authority["semantics"]
    assert semantics["contains_planned_timetable"] is False
    assert semantics["contains_planned_absolute_times"] is False
    assert semantics["contains_planned_trip_count"] is False
    assert semantics["contains_planned_trip_ids"] is False
    assert semantics["contains_expected_service_count"] is False
    assert semantics["service_count_must_be_inferred"] is True
    assert authority["absolute_physical_headway_floor_s"] == 90.0
    assert authority["normal_headway_margin_s"] == 20.0
    assert all(
        row["mapping_status"] == "UNRESOLVED_ANONYMOUS_STATION_MAPPING"
        and row["network_station_ids"] == ()
        for row in authority["depots"]
    )


def test_service_day_clock_conversion_uses_four_am_boundary():
    assert clock_to_service_seconds(4, 0) == 0
    assert clock_to_service_seconds(7, 30) == 3 * 3600 + 30 * 60
    assert clock_to_service_seconds(3, 59) == 23 * 3600 + 59 * 60


def test_verified_dense_floors_apply_only_inside_qualified_bands():
    authority = build_operating_authority()
    special = clock_to_service_seconds(8, 0)
    offpeak = clock_to_service_seconds(12, 0)

    floor, band = normal_headway_floor_for(
        authority,
        resource_id="B_SHARED_TRUNK",
        path_id="B_main",
        direction="Down",
        event_time_s=special,
    )
    assert floor == 130.0
    assert band == "B_SHARED_SPECIAL_AM"

    floor, band = normal_headway_floor_for(
        authority,
        resource_id="B_BRANCH_ONLY",
        path_id="B_branch",
        direction="Up",
        event_time_s=special,
    )
    assert floor == 430.0
    assert band == "B_BRANCH_SPECIAL_AM"

    floor, band = normal_headway_floor_for(
        authority,
        resource_id="A_MAIN",
        path_id="A_main",
        direction="Down",
        event_time_s=clock_to_service_seconds(7, 15),
    )
    assert floor == 160.0
    assert band == "A_PEAK_AM"

    floor, band = normal_headway_floor_for(
        authority,
        resource_id="C_MAIN",
        path_id="C_main",
        direction="Down",
        event_time_s=clock_to_service_seconds(17, 45),
    )
    assert floor == 130.0
    assert band == "C_PEAK_PM"

    floor, band = normal_headway_floor_for(
        authority,
        resource_id="A_MAIN",
        path_id="A_main",
        direction="Down",
        event_time_s=offpeak,
    )
    assert floor == 90.0
    assert band is None
