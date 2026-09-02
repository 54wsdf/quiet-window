# Agent Instructions

Scope: this entire repository.

## Repository role

This repository is a public execution worker for reproducible retrieval, validation, packaging, and transport of openly accessible transportation and mobility datasets.

It is **not** the canonical research repository, not a publication repository, and not a place for unpublished research content. Keep research interpretation, paper text, internal project decisions, private-source material, and canonical research registries elsewhere.

The README is intentionally generic. Do not expand it to describe the execution role or any private research context unless the repository owner explicitly asks.

Before adding or modifying acquisition workflows, read `.github/EXECUTION_POLICY.md`.

## Primary priorities

1. Prefer rail and urban/public transit datasets.
2. Acquire S/A-tier scientific assets even when they cover only one day, one case, or cannot yet be paired with another dataset.
3. When multiple strong sources exist, prioritize same-system and same-period combinations across schedule, actual operations, demand/OD, disruptions, infrastructure, rolling stock, and performance.
4. Where a year must be selected, prefer 2019 or later unless an older benchmark or historical series has clear scientific value.
5. Dataset size is not an exclusion criterion. Split, shard, paginate, or stream instead of dropping a valuable source.

## Public-repository boundary

Only public or explicitly authorized source material may be handled here.

Never commit or expose:

- credentials, tokens, cookies, API keys, passwords, OAuth material, rclone configuration, or service-account material;
- private repository contents or unpublished research artifacts;
- personal data that is not already lawfully published for the intended use;
- account-specific storage configuration unless the owner explicitly approves it for public disclosure.

Do not bypass authentication, access controls, paywalls, application requirements, anti-bot controls, or license restrictions. If a source requires a user identity, account registration, signed agreement, or manual approval, record it as requiring owner action rather than automating around it.

## Execution discipline

- Keep workflow permissions minimal; default to `contents: read`.
- Do not use `pull_request_target` for acquisition jobs.
- Do not grant forked PRs access to secrets or privileged actions.
- Prefer `workflow_dispatch` for expensive or bulk jobs.
- Never commit downloaded payloads to Git history.
- Use runner scratch space for raw acquisition and external persistent storage for durable copies.
- Use deterministic sharding for large files and record enough information to reassemble them exactly.
- Treat GitHub Artifacts as a transport buffer, not the canonical archive. Keep retention short when practical.
- A workflow is not considered complete merely because the job is green; verify payload bytes, manifest, checksum, and durable destination state.

## Provenance requirement

Every acquired logical asset must have a machine-readable manifest containing, when available:

- stable dataset ID;
- provider and source URL(s);
- retrieval timestamp in UTC;
- license or terms URL;
- source time coverage;
- original filename(s);
- byte size;
- SHA-256 checksum(s);
- shard/reassembly metadata if split;
- acquisition method and workflow/run provenance;
- final durable-storage classification/path supplied at execution time.

Prefer preserving provider-native raw files unchanged. Derived/processed data belongs in a separate processing layer, not mixed into raw acquisition output.

## Agent handoff

When resuming work, first inspect this file, `.github/EXECUTION_POLICY.md`, current workflow runs, and existing manifests. Do not assume a prior run completed successfully from its name or GitHub conclusion alone.
