# Agent Instructions

## Documentation

When making code changes, update the relevant documentation:

- **README.md** — Keep setup instructions, feature list, architecture diagram, and file table current
- **DEV.md** — Internal dev notes and decisions (don't restructure, append new notes)
- **src/agy_acp/AGENTS.md** — Package architecture: module map, import graph, design decisions. Update when adding, removing, renaming, or changing the responsibility of a module

If you add a new slash command, model, config option, or capability, make sure it's reflected in README.md.

## Tests

All tests must pass before declaring a change complete. Never dismiss a failing test as "unrelated" — investigate every failure. A broken main branch is an emergency: if a test failure is unrelated to the current work, suspend current work and fix main first before resuming.

## Dependencies

Never rely on transitive dependencies. Any package you directly import must be directly declared in `pyproject.toml`. If you add a new import, add the dependency.
