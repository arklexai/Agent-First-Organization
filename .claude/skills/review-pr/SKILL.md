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

Review a PR for functional correctness and project conventions. Focus on logic, architecture, and behavior. Formatting issues are caught by CI (ruff, pre-commit) and do not need manual review.

## Steps

### 1. Load PR context

```bash
gh pr view $ARGUMENTS --json title,body,files,additions,deletions
gh pr diff $ARGUMENTS
```

### 2. Validate PR title and description

- Title matches Conventional Commits regex, under 72 chars
- Description sections meet word minimums (Summary 5w, Description 10w, Tests 10w)
- These are also CI-enforced, so flag only if CI somehow missed them

### 3. Review functional changes

Focus on:
- Logic correctness: does the code do what the PR claims?
- Edge cases: are error paths handled?
- API changes: are they backwards compatible or properly marked as breaking?
- Security: no hardcoded secrets, no unsanitized input in dangerous contexts
- Performance: any obvious inefficiencies?

### 5. Check scope

- Single concern per PR
- Flag unrelated changes bundled together

### 6. Check test coverage

- New behavior should have tests
- Coverage minimum is 45%. Flag large additions without tests.

## Output

```
## Summary
One-sentence assessment.

## Findings
- List issues grouped by severity (blocking, suggestion, nit)

## Checklist
- [ ] Title matches Conventional Commits
- [ ] Description sections meet word minimums
- [ ] Functional correctness verified
- [ ] Tests cover new/changed code
- [ ] No secrets in diff
```

End with: APPROVED, CHANGES REQUESTED, or NEEDS DISCUSSION.
