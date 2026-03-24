---
name: pre-review
description: Check your branch before opening a PR (lint, format, tests, title)
disable-model-invocation: true
context: fork
agent: Explore
allowed-tools: Bash, Read, Grep, Glob
---

# Pre-review

Run local checks to catch issues before opening a PR.

## Steps

### 1. Check branch and changes

```bash
git branch --show-current
git status
git log --oneline main..HEAD
```

### 2. Run ruff linter

```bash
ruff check .
```

If there are fixable issues, run `ruff check --fix .` and report what was fixed.

### 3. Run ruff formatter

```bash
ruff format --check .
```

If files would be reformatted, run `ruff format .` and report what changed.

### 4. Run tests

```bash
pytest tests/ -x --tb=short
```

Report pass/fail count and any failures.

### 5. Check coverage

```bash
pytest tests/ --cov=arklex --cov-fail-under=45 --cov-report=term-missing -q
```

Coverage minimum is 45%. Flag if below threshold.

### 6. Validate branch name

Branch should follow `<type>/<short-description>` pattern, e.g. `feat/add-retry`.

### 7. Suggest PR title

Based on the commits on this branch, suggest a Conventional Commits title:
- Format: `<type>(<scope>): <description>`
- Max 72 characters
- Types: feat, fix, docs, chore, ci, build, refactor, test, perf, style, revert

### 8. Report

Summarize results:

```
Lint:     pass/fail
Format:   pass/fail
Tests:    X passed, Y failed
Coverage: XX% (minimum 45%)
Branch:   <name> (valid/invalid)
Title:    <suggested title>
```
