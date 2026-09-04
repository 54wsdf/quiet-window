import json
from pathlib import Path


def load_patch(path):
    patch = json.loads(Path(path).read_text(encoding="utf-8"))
    if patch.get("schema") != "mppd.seoul-topology-patch.g0c.v1":
        raise ValueError(f"unsupported topology patch schema: {patch.get('schema')}")
    edges = patch.get("edges") or []
    if not edges:
        raise ValueError("topology patch has no edges")
    return patch


def apply_topology_patch(G, meta, patch_or_path):
    patch = load_patch(patch_or_path) if isinstance(patch_or_path, (str, Path)) else patch_or_path
    records = []
    inserted = 0
    already_present = 0

    for spec in patch.get("edges") or []:
        edge_id = str(spec.get("edge_id") or "").strip()
        u = str(spec.get("u") or "").strip()
        v = str(spec.get("v") or "").strip()
        physical_station = str(spec.get("physical_station") or "").strip()
        evidence_class = str(spec.get("evidence_class") or "").strip()
        internal_evidence = list(spec.get("internal_evidence") or [])
        weight = float(spec.get("weight", 2.5))

        if not edge_id or not u or not v:
            raise ValueError(f"malformed patch edge: {spec}")
        if u == v:
            raise ValueError(f"self-loop topology patch forbidden: {edge_id}")
        if u not in G or v not in G:
            raise ValueError(f"patch edge references missing node: {edge_id}: {u}, {v}")
        if meta[u].get("line") == meta[v].get("line"):
            raise ValueError(f"same-line transfer patch forbidden: {edge_id}")
        if not physical_station or not evidence_class or len(internal_evidence) < 2:
            raise ValueError(f"patch edge lacks provenance evidence: {edge_id}")
        if weight <= 0:
            raise ValueError(f"patch edge has non-positive weight: {edge_id}")

        existed = G.has_edge(u, v)
        if existed:
            already_present += 1
            action = "ALREADY_PRESENT"
        else:
            G.add_edge(
                u,
                v,
                kind=str(spec.get("kind") or "transfer_repaired"),
                weight=weight,
                evidence=evidence_class,
                physical_station=physical_station,
                topology_patch_edge_id=edge_id,
                topology_patch_schema=patch.get("schema"),
                topology_patch_status=patch.get("status"),
                topology_patch_internal_evidence="|".join(map(str, internal_evidence)),
                observed_ats=False,
            )
            inserted += 1
            action = "INSERTED_FOR_QUALIFICATION"

        records.append(
            {
                "edge_id": edge_id,
                "u": u,
                "v": v,
                "line_u": meta[u].get("line"),
                "line_v": meta[v].get("line"),
                "physical_station": physical_station,
                "weight": weight,
                "evidence_class": evidence_class,
                "action": action,
            }
        )

    return {
        "schema": patch.get("schema"),
        "status": patch.get("status"),
        "edge_count_requested": len(patch.get("edges") or []),
        "edge_count_inserted": inserted,
        "edge_count_already_present": already_present,
        "records": records,
        "scientific_boundary": list(patch.get("scientific_boundary") or []),
    }
