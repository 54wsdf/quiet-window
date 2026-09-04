import json
from pathlib import Path

SUPPORTED_SCHEMAS = {
    "mppd.seoul-topology-patch.g0c.v1",
    "mppd.seoul-topology-patch.g0e.v1",
}
TRANSFER_KINDS = {"transfer_repaired"}
INLINE_KINDS = {"inline_gap_repaired", "shared_corridor_continuity_repaired"}


def load_patch(path):
    path = Path(path)
    patch = json.loads(path.read_text(encoding="utf-8"))
    schema = patch.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported topology patch schema: {schema}")

    inherited = None
    parent_name = str(patch.get("inherits_from") or "").strip()
    if parent_name:
        parent_path = Path(parent_name)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        inherited = load_patch(parent_path)

    own_edges = list(patch.get("edges") or [])
    parent_edges = list((inherited or {}).get("edges") or [])
    combined = parent_edges + own_edges
    if not combined:
        raise ValueError("topology patch has no edges")

    ids = [str(x.get("edge_id") or "") for x in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate topology patch edge_id across layered patch")

    expanded = dict(patch)
    expanded["edges"] = combined
    expanded["own_edge_count"] = len(own_edges)
    expanded["inherited_edge_count"] = len(parent_edges)
    expanded["expanded_edge_count"] = len(combined)
    if inherited:
        expanded["inherited_schema"] = inherited.get("schema")
        expanded["inherited_status"] = inherited.get("status")
    return expanded


def apply_topology_patch(G, meta, patch_or_path):
    patch = load_patch(patch_or_path) if isinstance(patch_or_path, (str, Path)) else patch_or_path
    records = []
    inserted = 0
    already_present = 0

    for spec in patch.get("edges") or []:
        edge_id = str(spec.get("edge_id") or "").strip()
        u = str(spec.get("u") or "").strip()
        v = str(spec.get("v") or "").strip()
        kind = str(spec.get("kind") or "").strip()
        physical_station = str(spec.get("physical_station") or "").strip()
        corridor = str(spec.get("corridor") or "").strip()
        evidence_class = str(spec.get("evidence_class") or "").strip()
        internal_evidence = list(spec.get("internal_evidence") or [])
        weight = float(spec.get("weight", 1.0))

        if not edge_id or not u or not v or not kind:
            raise ValueError(f"malformed patch edge: {spec}")
        if u == v:
            raise ValueError(f"self-loop topology patch forbidden: {edge_id}")
        if u not in G or v not in G:
            raise ValueError(f"patch edge references missing node: {edge_id}: {u}, {v}")
        if not evidence_class or len(internal_evidence) < 2:
            raise ValueError(f"patch edge lacks provenance evidence: {edge_id}")
        if weight <= 0:
            raise ValueError(f"patch edge has non-positive weight: {edge_id}")

        same_line = meta[u].get("line") == meta[v].get("line")
        if kind in TRANSFER_KINDS:
            if same_line:
                raise ValueError(f"transfer patch must be cross-line: {edge_id}")
            if not physical_station:
                raise ValueError(f"transfer patch lacks physical station: {edge_id}")
        elif kind in INLINE_KINDS:
            if not same_line:
                raise ValueError(f"continuity patch must be same-line: {edge_id}")
            if not corridor:
                raise ValueError(f"continuity patch lacks corridor label: {edge_id}")
            seq_u = meta[u].get("seq")
            seq_v = meta[v].get("seq")
            declared_gap = spec.get("sequence_gap")
            if seq_u is None or seq_v is None:
                raise ValueError(f"continuity patch endpoint lacks sequence: {edge_id}")
            actual_gap = abs(int(seq_u) - int(seq_v))
            if declared_gap is None or int(declared_gap) != actual_gap or actual_gap <= 1:
                raise ValueError(
                    f"continuity patch sequence gap mismatch: {edge_id}: declared={declared_gap}, actual={actual_gap}"
                )
        else:
            raise ValueError(f"unsupported topology patch edge kind: {edge_id}: {kind}")

        existed = G.has_edge(u, v)
        if existed:
            already_present += 1
            action = "ALREADY_PRESENT"
        else:
            attrs = {
                "kind": kind,
                "weight": weight,
                "evidence": evidence_class,
                "topology_patch_edge_id": edge_id,
                "topology_patch_schema": patch.get("schema"),
                "topology_patch_status": patch.get("status"),
                "topology_patch_internal_evidence": "|".join(map(str, internal_evidence)),
                "observed_ats": False,
            }
            if physical_station:
                attrs["physical_station"] = physical_station
            if corridor:
                attrs["corridor"] = corridor
            if spec.get("sequence_gap") is not None:
                attrs["sequence_gap"] = int(spec["sequence_gap"])
            G.add_edge(u, v, **attrs)
            inserted += 1
            action = "INSERTED_FOR_QUALIFICATION"

        records.append(
            {
                "edge_id": edge_id,
                "u": u,
                "v": v,
                "line_u": meta[u].get("line"),
                "line_v": meta[v].get("line"),
                "kind": kind,
                "physical_station": physical_station,
                "corridor": corridor,
                "weight": weight,
                "sequence_gap": spec.get("sequence_gap"),
                "evidence_class": evidence_class,
                "action": action,
            }
        )

    return {
        "schema": patch.get("schema"),
        "status": patch.get("status"),
        "inherits_from": patch.get("inherits_from"),
        "inherited_schema": patch.get("inherited_schema"),
        "own_edge_count": patch.get("own_edge_count", len(patch.get("edges") or [])),
        "inherited_edge_count": patch.get("inherited_edge_count", 0),
        "edge_count_requested": len(patch.get("edges") or []),
        "edge_count_inserted": inserted,
        "edge_count_already_present": already_present,
        "records": records,
        "scientific_boundary": list(patch.get("scientific_boundary") or []),
    }
