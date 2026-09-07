# Documentation is split three ways, and each part has one audience

- Status: adopted
- Area: all
- Date: 2026-09-07

## Decision

`docs/` and the rustdoc are for people using the library. They describe what a
thing is, how to call it, and what will surprise the caller. They carry no
record of what was measured, what was tried and rejected, or why the internals
are arranged as they are.

`decisions/` holds that record, one file per decision with an index in
`README.md`, so reading the index finds the one file that matters.

`CLAUDE.md` is loaded into context every session, so it holds only what is
needed before the task is known: where files live, what to touch when adding a
problem or a heuristic, the benchmark config shape, and this repo's
conventions. Everything an agent can read when it acts stays out.

Prose is plain. Bold marks a contract that fails silently, a few times across
the repository, plus the `**API:**` line each page carries. No em-dashes, and
colons only where one introduces a list or a code block.

## Measurement

The `CLAUDE.md` split was decided by what a session actually reached for. The
module map, the benchmark `kind` table and the site conventions were used
repeatedly and have no equivalent elsewhere. The per-algorithm table, the trait
descriptions and the `SearchState` narrative were never enough to act on, so the
work went to the source or to `docs/` every time. Removing them took the file
from 370 lines to 183 and left one identifier without a home, now 0013.

## Do not retry

Do not move measurements back into `docs/` or the rustdoc, and do not answer
"where should this note live" with `CLAUDE.md` by default.

A bulk find-and-replace over prose needs a review pass. Replacing every em-dash
mechanically produced comma splices, damaged six table cells that had held a
dash, and left eight lines starting with a stray comma where a dash had opened a
continuation line.
