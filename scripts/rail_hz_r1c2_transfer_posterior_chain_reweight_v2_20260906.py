from __future__ import annotations

from functools import lru_cache

import rail_hz_r1c2_transfer_posterior_chain_reweight_20260906 as base


def movement_station(movement: str) -> int:
    if movement.startswith("SAME_LINE_SERVICE_CHANGE:"):
        return int(movement.rsplit(":", 1)[1])
    if "->" in movement:
        left, right = movement.split("->", 1)
        a = int(left.rsplit(":", 1)[1])
        b = int(right.rsplit(":", 1)[1])
        if a != b:
            raise ValueError(f"movement uses unequal interchange station ids: {movement}")
        return a
    raise ValueError(f"cannot parse movement station: {movement}")


def parse_transfer_chain(raw: str, expected_count: int, known_movements) -> list[str] | None:
    known = tuple(sorted((str(x) for x in known_movements), key=lambda x: (-len(x), x)))

    @lru_cache(maxsize=None)
    def rec(pos: int, remain: int):
        if remain == 0:
            return () if pos == len(raw) else None
        for movement in known:
            if not raw.startswith(movement, pos):
                continue
            end = pos + len(movement)
            if remain == 1:
                if end == len(raw):
                    return (movement,)
                continue
            if end >= len(raw) or raw[end] != ">":
                continue
            tail = rec(end + 1, remain - 1)
            if tail is not None:
                return (movement,) + tail
        return None

    ans = rec(0, int(expected_count))
    return None if ans is None else list(ans)


def edge_log_ratio(row, old_events, old_median, old_sigma, by_key, root_meta, service_seq, fits):
    if str(row["descendant_state_type"]) == "UNRESOLVED" or int(row["transfer_count"]) <= 0:
        return 0.0, True, "NO_TRANSFER_FACTOR"
    roots = [x for x in str(row["root_chain"]).split(">") if x]
    expected = int(row["transfer_count"])
    movements = parse_transfer_chain(str(row["transfer_chain"]), expected, fits.keys())
    if movements is None:
        return 0.0, False, "TRANSFER_CHAIN_KNOWN_MOVEMENT_PARSE_FAILURE"
    if len(roots) != len(movements) + 1 or len(movements) != expected:
        return 0.0, False, "ROOT_TRANSFER_CHAIN_ARITY_MISMATCH_AFTER_STRUCTURED_PARSE"
    total = 0.0
    for i, movement in enumerate(movements):
        lower_root, upper_root = roots[i], roots[i + 1]
        new_p, new_reason = base.k1_boarded_probability(movement, lower_root, upper_root, by_key, root_meta, service_seq, fits)
        if new_p is None:
            return 0.0, False, "NEW:" + new_reason
        old_ll, old_reason = base.old_transfer_log_factor(movement, lower_root, upper_root, old_events, old_median, old_sigma)
        if old_ll is None:
            return 0.0, False, "OLD:" + old_reason
        total += base.math.log(max(new_p, base.EPS)) - old_ll
    return total, True, "OK"


base.movement_station = movement_station
base.edge_log_ratio = edge_log_ratio


if __name__ == "__main__":
    base.main()
