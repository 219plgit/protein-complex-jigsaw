# Protein Complex Jigsaw v4.0.0

This is the first release under the public name **Protein Complex Jigsaw (PCJ)**. It follows the historical `scijigsaw` v3.x line; the major-version increment avoids resetting or obscuring the existing software release history.

## Naming and compatibility

- Preferred project name: **Protein Complex Jigsaw (PCJ)**.
- Preferred Python namespace: `pcjigsaw`.
- Distribution name: `protein-complex-jigsaw`.
- Preferred CLI prefix: `pcjigsaw-`.
- Repository: `219plgit/protein-complex-jigsaw`.
- Historical `scijigsaw` imports and CLI commands remain available as compatibility aliases.

## Variant-to-mechanism analysis

Adds an explicit perturbation layer for molecular effects supplied by the user or an upstream variant-effect method:

- component unavailability;
- lost interfaces;
- added exclusions/competition;
- recomputation of admissible seed-anchored and merger-tree spaces;
- productive-target reachability;
- largest reachable/trapped state;
- first blocked transitions;
- unavailable interfaces;
- surviving productive-route flag;
- JSON-serialisable reports.

The module does not infer pathogenicity from sequence.

## Failure and misassembly

Adds an evidence-gated misassembly layer. Productive assembly failure is reported independently of aggregation. Off-pathway outcomes are returned only when an explicit provenance-bearing rule is supplied.

## Existing capabilities retained

The deterministic exact-enumeration core, typed relations, assembly-tree enumeration, support queries, structure extraction, rendering, educational tiles, probabilistic extensions and manuscript reproduction scripts remain available without forcing changes to existing `scijigsaw` analyses.
