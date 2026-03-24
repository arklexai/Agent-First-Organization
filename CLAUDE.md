# Arklex - Agent-First Organization

Open-source agent-first framework for building AI agents. Python 3.10+, MIT license.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[faiss,llm]"
pip install pre-commit && pre-commit install
```

## Code style

- Formatter/linter: **Ruff** (format + check), enforced by pre-commit hooks
- Line length: 88
- Lint rules: E, F, I, UP, ANN, B, SIM, C4, TID (E501 ignored)
- Format: double quotes, space indent, auto line endings
- Absolute imports only; type annotations on all function signatures
- No license header required (MIT)

## PR conventions

- **Title**: Conventional Commits format, max 72 chars
  - `<type>(<scope>): <description>`
  - Types: feat, fix, docs, chore, ci, build, refactor, test, perf, style, revert
  - Scope is optional but recommended (lowercase, may contain digits/hyphens/underscores)
  - Common mistakes: using component name as type (`orchestrator: ...`), ticket ID as type
- **Description**: use the PR template. CI enforces minimum word counts:
  - Summary: 5 words, Description: 10 words, Tests: 10 words
- **Merge strategy**: squash merge

## Testing

```bash
pytest tests/                          # all tests
pytest tests/ -m "not integration"     # skip integration
pytest tests/ -m shopify               # specific marker
pytest tests/ --cov=arklex --cov-fail-under=45  # with coverage gate
```

- Test paths: `tests/` (subdirs: data, env, orchestrator, temp, utils)
- Coverage minimum: **45%** (enforced in CI)
- Markers: integration, no_intent_mock, no_llm_mock, shopify, hubspot, hitl, slow
- Async mode: auto (pytest-asyncio)

### Test labels (CI)

Add one of these labels to your PR to trigger the appropriate test workflow:
- `run-coverage-tests` - full coverage run
- `run-diff-coverage-tests` - coverage on changed files only
- `run-integration-tests` - integration test suite (skips coverage)

## CI checks

- `pr-check.yml` validates PR description sections (Summary, Description, Tests)
- PR title validation enforces Conventional Commits format and 72-char limit
- Coverage workflows triggered by PR labels (see above)

## Pre-commit hooks

Configured in `.pre-commit-config.yaml`:
- `ruff --fix` (auto-fix lint issues)
- `ruff-format` (auto-format)

## Docs

Documentation uses **Docusaurus**, located in `docs/`. Config: `docs/docusaurus.config.js`.

## Writing style

- Never use em dashes or en dashes in prose. Use hyphens only where grammatically needed.
- Rephrase sentences instead of inserting dashes as punctuation.

## Branch naming

Use `<type>/<short-description>`, e.g. `feat/add-retry-logic`, `fix/null-pointer`.
