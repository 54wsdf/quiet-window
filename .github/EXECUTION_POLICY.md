# Execution Policy

This file defines the operating discipline for acquisition workflows in this repository.

## 1. Intended workload

The repository may run public-data acquisition, validation, checksum, pagination, decompression/repacking, deterministic splitting, and export/transport jobs for transportation research assets.

Preferred asset classes, roughly in order:

- rail/public-transit actual operations and historical archives;
- timetable/GTFS/NeTEx and other machine-readable service plans;
- AFC, OD, ridership, passenger-flow and occupancy data;
- disruption, delay, cancellation, incident and service-alert data;
- infrastructure, track layout, capacity and work/closure restrictions;
- rolling-stock composition, circulation, depot, shunting and maintenance data;
- high-quality railway scheduling/dispatching/shunting benchmarks;
- high-value road/trajectory/mobility assets when scientifically useful.

S/A-tier sources are worth acquiring even if isolated in time. Temporal matching increases priority but is not a prerequisite.

## 2. Source and terms checks

Before acquisition, record the provider page and terms/license URL when available.

Allowed by default:

- public government/open-data downloads;
- public transport-operator open data;
- public APIs whose documented use permits automated retrieval;
- public GitHub/Zenodo/Figshare/UCI-style research assets under compatible terms;
- openly downloadable benchmark instances and reproducibility packages.

Do not automate around sources requiring login, application approval, signed agreements, paywalls, identity verification, or technical access controls. Those must be surfaced for owner action.

## 3. Time selection

When a source requires choosing a year or period:

- prefer 2019 onward;
- retain newer complete years and current partial years where useful;
- keep older years when needed for a benchmark, long-run comparison, disruption case, or historically unique asset;
- for frequently published schedule snapshots, avoid mechanically storing near-identical duplicates unless version history itself is scientifically useful;
- for actual operations, disruption, demand and other realized data, preserve the finest practical native temporal granularity.

## 4. Large-data handling

There is no scientific byte-size cutoff.

Engineering limits must be handled using one or more of:

- API pagination;
- date/month/year sharding;
- deterministic byte splitting;
- resumable or ranged download where supported;
- parallel jobs with bounded concurrency;
- source-native partitions such as Parquet day partitions.

Never silently truncate a dataset to satisfy runner, API, or artifact constraints.

For byte-split payloads, preserve:

- original filename;
- original byte length;
- original SHA-256;
- ordered part names;
- byte length and SHA-256 for every part;
- exact reassembly command or algorithm.

## 5. Suggested repository layout

Create directories only when needed; do not add empty decorative structure.

```text
.github/workflows/        # execution entry points
scripts/lib/              # reusable download/checksum/split helpers
scripts/fetchers/         # provider-specific resolvers
configs/sources/          # public source definitions only
schemas/                  # manifest/source schemas
```

Downloaded data, temporary shards and credentials must remain outside Git history.

## 6. Workflow conventions

- Use concise neutral workflow names that describe the technical job accurately.
- Prefer one logical source family per workflow or reusable workflow interface.
- Expensive harvest jobs should normally use `workflow_dispatch`.
- Set explicit `timeout-minutes` for network jobs.
- Use `fail-fast: false` for independent matrix acquisitions.
- Use retry/backoff for transient network failures.
- Capture HTTP failures and zero-byte/HTML-error payloads as failures, not successful datasets.
- Validate content type or archive readability when possible.
- Generate SHA-256 before declaring acquisition success.
- Avoid writing source data into job logs.
- Do not print secret-bearing headers, signed URLs, cookies or credentials.

Default permission block:

```yaml
permissions:
  contents: read
```

Escalate permissions only for a specific documented step.

## 7. Artifact discipline

GitHub Artifacts are temporary transport objects.

- Do not treat an Artifact as the canonical copy.
- Prefer short retention after durable storage has been verified.
- Avoid duplicating the same multi-GB source across multiple Artifacts.
- When an existing Artifact contains the verified source bytes, split/repackage it instead of redownloading the provider source.
- If Artifact quota is exhausted, stop creating new Artifacts and prioritize durable transfer or a direct-to-storage path.

## 8. Durable path model

The persistent archive path is supplied by the controlling workflow/context and must follow this semantic model when possible:

```text
transport_data_lake/
  01_rail/{system}/
    00_source_archives/
    01_schedule/
    02_actual_operations/
    03_realtime/
    04_demand_od/
    05_disruptions_trackwork/
    06_infrastructure/
    07_rolling_stock_composition/
    08_performance_reliability/
    99_manifests/
```

Other modes may use `02_road`, `03_trajectory`, `04_mobility_demand`, and `05_simulation_benchmarks`.

Do not hard-code private storage credentials in this repository. Storage IDs/paths should be injected at run time unless explicitly approved as public metadata.

## 9. Completion gates

A logical asset is `ACQUIRED` only when:

1. provider bytes were actually obtained;
2. size is non-zero and content is plausible;
3. SHA-256/manifest exists;
4. all shards required for reconstruction exist;
5. durable-storage transfer is confirmed.

Use intermediate states such as `DISCOVERED`, `DOWNLOADED_TEMP`, `ARTIFACT_ONLY`, `TRANSFER_PARTIAL`, `ACQUIRED`, and `FAILED` rather than collapsing all green jobs into success.

## 10. Public-repository hygiene

This repository is public. Assume every committed byte and every Actions log is visible to third parties.

Do not include private research rationale, unpublished hypotheses, paper drafts, private repository names/paths when avoidable, internal credentials, personal storage identifiers, or user-specific sensitive metadata.

The repository can describe its technical execution requirements to agents without advertising private research context in README.
