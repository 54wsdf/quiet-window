from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

SERVICE_DAY_BOUNDARY_HOUR = 4
ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S = 90.0
NORMAL_HEADWAY_MARGIN_S = 20.0

OFFICIAL_20181022_URL = (
    "https://hznews.hangzhou.com.cn/chengshi/content/2018-10/22/content_7084512.htm"
)
SECONDARY_20181022_URL = (
    "https://www.train-times.net/article/hangzhou20181022-html"
)


@dataclass(frozen=True)
class OperatingHeadwayBand:
    band_id: str
    resource_id: str
    path_ids: tuple[str, ...]
    station_ids: tuple[int, ...]
    directions: tuple[str, ...]
    start_service_s: int
    end_service_s: int
    observed_dense_headway_s: float
    normal_service_floor_s: float
    source_url: str
    source_role: str
    evidence_note: str

    def __post_init__(self) -> None:
        if not self.band_id or not self.resource_id:
            raise ValueError("headway band IDs must be non-empty")
        if not self.path_ids or not self.station_ids or not self.directions:
            raise ValueError(f"{self.band_id}: resource scope must be non-empty")
        if self.start_service_s < 0 or self.end_service_s <= self.start_service_s:
            raise ValueError(f"{self.band_id}: invalid service-day time window")
        if self.observed_dense_headway_s < ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S:
            raise ValueError(f"{self.band_id}: observed dense headway below physical floor")
        expected = max(
            ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
            self.observed_dense_headway_s - NORMAL_HEADWAY_MARGIN_S,
        )
        if abs(self.normal_service_floor_s - expected) > 1e-9:
            raise ValueError(
                f"{self.band_id}: normal floor must equal max(90, observed_dense-20)"
            )


@dataclass(frozen=True)
class DepotAuthorityRecord:
    depot_id: str
    public_name: str
    line_ids: tuple[str, ...]
    role: str
    source_role: str
    network_station_ids: tuple[int, ...] = ()
    mapping_status: str = "UNRESOLVED_ANONYMOUS_STATION_MAPPING"

    def __post_init__(self) -> None:
        if not self.depot_id or not self.public_name or not self.line_ids:
            raise ValueError("depot authority records require ID, name and line scope")
        if self.mapping_status == "UNRESOLVED_ANONYMOUS_STATION_MAPPING" and self.network_station_ids:
            raise ValueError("unresolved depot mapping may not claim anonymous station IDs")


def clock_to_service_seconds(hour: int, minute: int = 0, second: int = 0) -> int:
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        raise ValueError("invalid wall-clock time")
    clock_s = hour * 3600 + minute * 60 + second
    boundary_s = SERVICE_DAY_BOUNDARY_HOUR * 3600
    if clock_s < boundary_s:
        clock_s += 24 * 3600
    return clock_s - boundary_s


def _band(
    band_id: str,
    resource_id: str,
    path_ids: Sequence[str],
    station_ids: Sequence[int],
    start_hm: tuple[int, int],
    end_hm: tuple[int, int],
    observed_dense_headway_s: float,
    *,
    source_url: str = SECONDARY_20181022_URL,
    evidence_note: str,
) -> OperatingHeadwayBand:
    return OperatingHeadwayBand(
        band_id=band_id,
        resource_id=resource_id,
        path_ids=tuple(path_ids),
        station_ids=tuple(int(x) for x in station_ids),
        directions=("Down", "Up"),
        start_service_s=clock_to_service_seconds(*start_hm),
        end_service_s=clock_to_service_seconds(*end_hm),
        observed_dense_headway_s=float(observed_dense_headway_s),
        normal_service_floor_s=max(
            ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
            float(observed_dense_headway_s) - NORMAL_HEADWAY_MARGIN_S,
        ),
        source_url=source_url,
        source_role="PUBLIC_OPERATING_DENSITY_ENVELOPE_ONLY",
        evidence_note=evidence_note,
    )


def build_operating_authority() -> dict[str, Any]:
    """Return a plan-free operating-envelope authority for 2019-01-04 R1.

    The authority constrains physically/operationally admissible service density but
    contains no planned absolute event times, trip inventory, train IDs or target
    service count.  It is therefore eligible for COUNT_FREE structural qualification.

    The 2018-10-22 public operating notice gives the workday peak windows.  A
    timetable-analysis source reports the dense intervals after that revision.
    Where the exact sub-window of a reported dense interval is not independently
    resolved, the implementation deliberately applies only the weakest verified
    dense floor across the broader peak window rather than inventing a tighter one.
    """

    special_windows = (((7, 30), (8, 30)), ((17, 30), (18, 30)))
    peak_windows = (((7, 0), (9, 0)), ((17, 0), (19, 0)))
    bands: list[OperatingHeadwayBand] = []

    # Line B = historical Line 1.  The shared trunk is stations 0..20.  The
    # 2018-10-22 revision reports a 150 s dense interval on the shared/main service
    # and 450 s on the Linping branch.  Only the special-peak hour is hard-qualified
    # at these dense floors here; other periods retain the universal 90 s floor
    # until a source-exact interval table is encoded.
    for idx, (start, end) in enumerate(special_windows):
        suffix = "AM" if idx == 0 else "PM"
        bands.append(
            _band(
                f"B_SHARED_SPECIAL_{suffix}",
                "B_SHARED_TRUNK",
                ("B_main", "B_branch"),
                range(0, 21),
                start,
                end,
                150.0,
                evidence_note="2018-10-22 dense shared-trunk interval reported as 2m30s",
            )
        )
        bands.append(
            _band(
                f"B_BRANCH_SPECIAL_{suffix}",
                "B_BRANCH_ONLY",
                ("B_branch",),
                range(28, 34),
                start,
                end,
                450.0,
                evidence_note="2018-10-22 Linping-branch dense interval reported as 7m30s",
            )
        )

    # Line C = historical Line 2.  The revision retained a 150 s densest peak
    # interval but shortened its duration; the precise 150/230 s sub-window split
    # is not source-exact in the textual evidence used here.  Use the conservative
    # 130 s normal floor throughout the documented peak windows, never the stronger
    # 210 s floor unless a source-exact sub-window authority is added later.
    for idx, (start, end) in enumerate(peak_windows):
        suffix = "AM" if idx == 0 else "PM"
        bands.append(
            _band(
                f"C_PEAK_{suffix}",
                "C_MAIN",
                ("C_main",),
                range(34, 67),
                start,
                end,
                150.0,
                evidence_note=(
                    "2018-10-22 revision retained 2m30s densest peak running while "
                    "some peak periods widened to 3m50s; exact sub-window unresolved"
                ),
            )
        )

    # Line A = historical Line 4 phase 1.  The 2018-10-22 revision reports 180 s
    # workday peak running.  This gives a 160 s normal-service qualification floor.
    a_nodes = (67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16)
    for idx, (start, end) in enumerate(peak_windows):
        suffix = "AM" if idx == 0 else "PM"
        bands.append(
            _band(
                f"A_PEAK_{suffix}",
                "A_MAIN",
                ("A_main",),
                a_nodes,
                start,
                end,
                180.0,
                evidence_note="2018-10-22 workday peak interval reported as 3m00s",
            )
        )

    depots = (
        DepotAuthorityRecord(
            depot_id="HZ_L1_QIBAO",
            public_name="Qibao Depot",
            line_ids=("B", "A"),
            role="DEPOT",
            source_role="PUBLIC_DEPOT_NAME_AND_LINE_ASSOCIATION_ONLY",
        ),
        DepotAuthorityRecord(
            depot_id="HZ_L1_XIANGHU",
            public_name="Xianghu Parking Yard",
            line_ids=("B",),
            role="PARKING_YARD",
            source_role="PUBLIC_DEPOT_NAME_AND_LINE_ASSOCIATION_ONLY",
        ),
        DepotAuthorityRecord(
            depot_id="HZ_L2_SHUSHAN",
            public_name="Shushan Depot",
            line_ids=("C",),
            role="DEPOT",
            source_role="PUBLIC_DEPOT_NAME_AND_LINE_ASSOCIATION_ONLY",
        ),
        DepotAuthorityRecord(
            depot_id="HZ_L2_SHUANGQIAO",
            public_name="Shuangqiao Parking Yard",
            line_ids=("C",),
            role="PARKING_YARD",
            source_role="PUBLIC_DEPOT_NAME_AND_LINE_ASSOCIATION_ONLY",
        ),
    )

    return {
        "schema": "mppd.r1-hz-operating-authority.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "service_date_scope": "2019-01-04_WORKDAY",
        "service_day_boundary_hour": SERVICE_DAY_BOUNDARY_HOUR,
        "absolute_physical_headway_floor_s": ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
        "normal_headway_margin_s": NORMAL_HEADWAY_MARGIN_S,
        "headway_bands": [asdict(x) for x in bands],
        "depots": [asdict(x) for x in depots],
        "sources": {
            "official_peak_windows": OFFICIAL_20181022_URL,
            "dense_interval_analysis": SECONDARY_20181022_URL,
        },
        "semantics": {
            "contains_planned_timetable": False,
            "contains_planned_absolute_times": False,
            "contains_planned_trip_count": False,
            "contains_planned_trip_ids": False,
            "contains_expected_service_count": False,
            "contains_operational_headway_envelope": True,
            "contains_absolute_physical_headway_floor": True,
            "depot_network_station_mapping_resolved": False,
            "service_count_must_be_inferred": True,
            "count_free_primary_inference_eligible": True,
        },
    }


def validate_operating_authority(authority: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    semantics = authority.get("semantics")
    if not isinstance(semantics, Mapping):
        return ["missing semantics"]
    forbidden = (
        "contains_planned_timetable",
        "contains_planned_absolute_times",
        "contains_planned_trip_count",
        "contains_planned_trip_ids",
        "contains_expected_service_count",
    )
    for key in forbidden:
        if semantics.get(key) is not False:
            errors.append(f"{key} must be false for COUNT_FREE authority")
    if semantics.get("service_count_must_be_inferred") is not True:
        errors.append("service_count_must_be_inferred must be true")
    if semantics.get("count_free_primary_inference_eligible") is not True:
        errors.append("count_free_primary_inference_eligible must be true")
    if float(authority.get("absolute_physical_headway_floor_s", -1)) != 90.0:
        errors.append("absolute physical headway floor must be 90 s")
    if float(authority.get("normal_headway_margin_s", -1)) != 20.0:
        errors.append("normal headway margin must be 20 s")
    for row in authority.get("depots", ()):  # unresolved mappings must fail closed
        if not isinstance(row, Mapping):
            errors.append("depot row must be mapping")
            continue
        if row.get("mapping_status") == "UNRESOLVED_ANONYMOUS_STATION_MAPPING" and row.get(
            "network_station_ids"
        ):
            errors.append(f"{row.get('depot_id')}: unresolved depot claims station IDs")
    return errors


def normal_headway_floor_for(
    authority: Mapping[str, Any],
    *,
    resource_id: str,
    path_id: str,
    direction: str,
    event_time_s: float,
) -> tuple[float, str | None]:
    """Return the strongest source-qualified floor active at this event time."""

    floor = float(authority["absolute_physical_headway_floor_s"])
    source_band: str | None = None
    for raw in authority.get("headway_bands", ()):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("resource_id")) != resource_id:
            continue
        if path_id not in tuple(str(x) for x in raw.get("path_ids", ())):
            continue
        if direction not in tuple(str(x) for x in raw.get("directions", ())):
            continue
        if float(raw.get("start_service_s", -1)) <= event_time_s < float(
            raw.get("end_service_s", -1)
        ):
            candidate = float(raw["normal_service_floor_s"])
            if candidate > floor:
                floor = candidate
                source_band = str(raw["band_id"])
    return floor, source_band
