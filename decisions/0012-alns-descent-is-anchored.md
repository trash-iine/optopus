# The descent after each ruin is anchored at the re-inserted customers

- Status: adopted
- Area: vrp
- Date: 2026-08
- Code: src/heuristic/specific/vrp/alns.rs

## Decision

Each recreated solution runs through `ops::Descent::run_around`, anchored at the
re-inserted customers plus their five nearest partners, rather than a full sweep
over every customer.

## Measurement

Anchored, over ten X instances at 30s by five runs, it is worth −0.38% mean
objective with 8 of 10 improved, and −0.97% on X-n701 at 60s.

A full sweep per iteration was +0.07%, a wash, and −2% on X-n459. On a mid-sized
instance the ruin's anchors widened by a full candidate list already cover
everything, so the extra work buys nothing.

## Do not retry

Do not replace the anchored descent with a full sweep for simplicity. The
anchoring is what makes the descent pay here.
