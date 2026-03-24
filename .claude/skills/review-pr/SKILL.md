---
name: review-pr
description: Review a pull request against project standards
disable-model-invocation: true
effort: high
argument-hint: "[PR-number]"
context: fork
agent: Explore
allowed-tools: Bash, Read, Grep, Glob
---

# Review PR

Review a PR for code quality, conventions, and CI readiness.

## Steps

### 1. Load PR context

```bash
gh pr view "$1" --json title,body,files,labels,additions,deletions
gh pr diff "$1"
```

### 2. Validate PR title

Check against Conventional Commits regex (must pass CI):

```
^(feat|fix|docs|chore|ci|build|refactor|test|perf|style|revert)(\([a-z][a-z0-9_-]*\))?!?: .+$
```

- Max 72 characters
- Common mistakes: component name as type (`orchestrator: ...`), ticket ID as type

### 3. Validate PR description

Check the PR body against the template in `.github/pull_request_template.md`:
- `## Summary` section: at least 5 words of content
- `## Description` section: at least 10 words of content
- `## Tests` section: at least 10 words of content
- Comments and unchecked checkboxes do not count

### 4. Check test labels

Verify one of these labels is present:
- `run-coverage-tests`
- `run-diff-coverage-tests`
- `run-integration-tests`

If missing, flag it. The PR template asks contributors to attach one.

### 5. Review code changes

For each changed file in `gh pr diff`:
- **Python files**: check for type annotations, absolute imports, ruff compliance
- **Test files**: verify new/changed code has corresponding tests
- Coverage minimum is 45%. Flag large additions without tests.
- No license headers needed (MIT)

### 6. Check for common issues

- Secrets or credentials in diff (API keys, tokens, passwords)
- Large files or binaries that should not be committed
- Changes to .env files (should be .env.example only)
- Missing docstrings on public functions/classes

### 7. Write review

Structure the review as:

```
## Summary
One-sentence summary of the PR.

## Findings
- List issues grouped by severity (blocking, suggestion, nit)

## Checklist
- [ ] Title matches Conventional Commits
- [ ] Description sections meet word minimums
- [ ] Test label attached
- [ ] Tests cover new/changed code
- [ ] No secrets in diff
```

Use natural sentence breaks. Do not use em dashes or en dashes.
