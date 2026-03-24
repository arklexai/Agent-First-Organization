---
name: draft-pr
description: Generate a PR title and description from your changes
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

# Draft PR

Generate a PR title and description that will pass CI checks.

## Steps

### 1. Gather context

```bash
git branch --show-current
git log --oneline main..HEAD
git diff main..HEAD --stat
gh pr list --head "$(git branch --show-current)" --state=open --json url -q '.[0].url'
```

### 2. Read PR template

Read `.github/pull_request_template.md` for the required structure.

### 3. Generate PR title

- Format: `<type>(<scope>): <description>` (Conventional Commits)
- Max 72 characters
- Types: feat, fix, docs, chore, ci, build, refactor, test, perf, style, revert
- Scope is optional but recommended

### 4. Generate PR description

Fill in each section from the template:

- **Summary** (min 5 words): concise overview of what the PR does
- **Description** (min 10 words): why the change is needed, how it works, side effects
- **Tests** (min 10 words): describe what was tested, mention test commands run

### 5. Remind about test labels

The contributor must attach one of these labels after creating the PR:
- `run-coverage-tests` - full test suite with coverage (45% minimum)
- `run-diff-coverage-tests` - coverage on changed files only
- `run-integration-tests` - integration tests (skips coverage gate)

### 6. Output

Print the generated title and full PR body in raw markdown format, ready to copy.
If an open PR already exists for this branch, offer to update it with `gh pr edit`.
