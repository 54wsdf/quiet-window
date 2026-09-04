import json
from collections import Counter
from pathlib import Path


def load_overlay(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "mppd.seoul-gtxa-crosswalk-overlay.g0h.v1":
        raise ValueError(f"unsupported GTX-A overlay schema: {payload.get('schema')}")
    if str(payload.get("business_date")) != "2026-08-29":
        raise ValueError("GTX-A overlay is date-specific and must match 2026-08-29")
    return payload


def _name_match(meta_row, aliases):
    vals = {
        str(meta_row.get("dv_name") or "").strip(),
        str(meta_row.get("service_name") or "").strip(),
    }
    vals.discard("")
    return bool(vals & {str(x).strip() for x in aliases if str(x).strip()})


def apply_gtxa_overlay(G, meta, code_to_nodes, overlay_or_path):
    overlay = load_overlay(overlay_or_path) if isinstance(overlay_or_path, (str, Path)) else overlay_or_path
    stations = {str(x["code"]): x for x in overlay.get("stations", [])}
    component_by_code = {}
    for comp in overlay.get("operational_components", []):
        component_id = str(comp.get("component_id") or "").strip()
        for code in comp.get("stations_in_service_order", []):
            component_by_code[str(code)] = component_id

    inserted_nodes = []
    reused_nodes = []
    code_overrides = []
    removed_inactive_codes = []
    inline_records = []
    transfer_records = []
    unresolved_transfer_targets = []
    resolved_transfer_count_by_component = Counter()

    for code, spec in stations.items():
        node = str(spec.get("node") or "").strip()
        active = bool(spec.get("active"))
        if not active:
            if code in code_to_nodes:
                removed_inactive_codes.append({"code": code, "old_nodes": list(code_to_nodes[code])})
                code_to_nodes.pop(code, None)
            continue
        if not node:
            raise ValueError(f"active GTX-A station lacks node id: {code}")
        if node not in G:
            station_id = node.split("|", 1)[1] if "|" in node else node
            m = {
                "line": "1032",
                "station": station_id,
                "seq": int(spec["sequence"]),
                "dv_name": str(spec.get("name") or "").strip(),
                "service_name": str(spec.get("name") or "").strip(),
                "tier": "GTXA_G0H_LINE_AWARE_OVERLAY",
                "overlay_station_code": code,
                "overlay_evidence": "G0G_RAW_AFC_PLUS_OFFICIAL_GTXA_SEQUENCE",
            }
            meta[node] = m
            G.add_node(node, **m)
            inserted_nodes.append(node)
        else:
            reused_nodes.append(node)
            meta[node]["seq"] = int(spec["sequence"])
            G.nodes[node]["seq"] = int(spec["sequence"])
            G.nodes[node]["overlay_station_code"] = code
            G.nodes[node]["overlay_evidence"] = "G0G_RAW_AFC_PLUS_OFFICIAL_GTXA_SEQUENCE"
        old = list(code_to_nodes.get(code, []))
        code_to_nodes[code] = [node]
        code_overrides.append({
            "code": code,
            "old_nodes": old,
            "new_node": node,
            "mapping_action": spec.get("mapping_action"),
            "operational_component_id": component_by_code.get(code),
        })

    for e in overlay.get("inline_edges", []):
        uc = str(e["u_code"]); vc = str(e["v_code"])
        if uc not in code_to_nodes or vc not in code_to_nodes:
            raise ValueError(f"GTX-A inline edge endpoint inactive/unmapped: {uc}-{vc}")
        u = code_to_nodes[uc][0]; v = code_to_nodes[vc][0]
        if meta[u]["line"] != "1032" or meta[v]["line"] != "1032":
            raise ValueError(f"GTX-A inline edge left line 1032: {uc}-{vc}")
        if G.has_edge(u, v):
            action = "ALREADY_PRESENT"
        else:
            G.add_edge(
                u, v,
                kind="gtxa_date_aware_inline",
                weight=float(e.get("weight", 1.0)),
                line="1032",
                component_id=e.get("component_id"),
                evidence="G0H_DATE_AWARE_GTXA_OPERATIONAL_COMPONENT",
                observed_ats=False,
            )
            action = "INSERTED"
        inline_records.append({"u_code": uc, "v_code": vc, "u": u, "v": v, "action": action, "weight": float(e.get("weight", 1.0)), "component_id": e.get("component_id")})

    for code, spec in stations.items():
        if not spec.get("active"):
            continue
        source = code_to_nodes[code][0]
        component_id = component_by_code.get(code)
        for target in list(spec.get("transfer_targets", [])):
            line = str(target["line"])
            aliases = list(target.get("station_names") or [])
            matches = sorted(
                n for n, m in meta.items()
                if n in G and m.get("line") == line and _name_match(m, aliases)
            )
            if not matches:
                unresolved_transfer_targets.append({
                    "code": code,
                    "source": source,
                    "operational_component_id": component_id,
                    "target_line": line,
                    "aliases": aliases,
                    "action": "UNRESOLVED_IN_BASE_REPRESENTATION",
                    "scientific_boundary": "The target line/station is not explicit in the current coarse P1C/G0E graph; no synthetic transfer is guessed."
                })
                continue
            for target_node in matches:
                if source == target_node:
                    continue
                if G.has_edge(source, target_node):
                    action = "ALREADY_PRESENT"
                else:
                    G.add_edge(
                        source, target_node,
                        kind="gtxa_transfer_repaired",
                        weight=2.5,
                        physical_station=str(spec.get("name") or ""),
                        evidence="G0H_LINE_AWARE_GTXA_TRANSFER_OVERLAY",
                        observed_ats=False,
                    )
                    action = "INSERTED"
                if component_id:
                    resolved_transfer_count_by_component[component_id] += 1
                transfer_records.append({
                    "code": code,
                    "source": source,
                    "operational_component_id": component_id,
                    "target_line": line,
                    "target_node": target_node,
                    "aliases": aliases,
                    "action": action,
                })

    active_component_ids = [
        str(comp.get("component_id") or "").strip()
        for comp in overlay.get("operational_components", [])
        if str(comp.get("status") or "").upper() == "ACTIVE"
    ]
    missing_component_attachment = [
        component_id for component_id in active_component_ids
        if resolved_transfer_count_by_component[component_id] == 0
    ]
    if missing_component_attachment:
        raise ValueError(
            "GTX-A operating component has no resolvable transfer attachment to the current base graph: "
            + ",".join(missing_component_attachment)
        )

    north = [code_to_nodes[c][0] for c in ["9000", "9001", "9002", "9004", "9005"]]
    south = [code_to_nodes[c][0] for c in ["9007", "9008", "9009", "9010"]]
    if any(G.has_edge(u, v) for u in north for v in south):
        raise RuntimeError("GTX-A north/south components acquired a direct forbidden edge")

    return {
        "schema": overlay.get("schema"),
        "status": overlay.get("status"),
        "business_date": overlay.get("business_date"),
        "inserted_node_count": len(inserted_nodes),
        "inserted_nodes": inserted_nodes,
        "reused_node_count": len(reused_nodes),
        "reused_nodes": reused_nodes,
        "code_override_count": len(code_overrides),
        "code_overrides": code_overrides,
        "removed_inactive_codes": removed_inactive_codes,
        "inline_edge_records": inline_records,
        "transfer_edge_records": transfer_records,
        "unresolved_transfer_targets": unresolved_transfer_targets,
        "resolved_transfer_count_by_component": dict(resolved_transfer_count_by_component),
        "all_active_components_have_base_network_attachment": not missing_component_attachment,
        "north_active_nodes": north,
        "south_active_nodes": south,
        "north_south_direct_edge_forbidden": True,
        "scientific_boundary": list(overlay.get("scientific_boundary") or []) + [
            "Declared transfer targets absent from the coarse base graph are retained as unresolved provenance records and are never guessed into existence.",
            "Qualification requires each active GTX-A operating component to have at least one explicitly resolved attachment to the existing network; it does not require every transfer station to be explicit in the coarse graph."
        ],
    }
