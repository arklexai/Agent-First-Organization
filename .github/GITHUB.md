# GitHub Actions and Workflows

This directory contains GitHub Actions and Workflows for CI/CD automation.

## Directory Structure

```
.github/
├── actions/           # Composite Actions (reusable steps)
├── workflows/         # Reusable Workflows and Event-triggered Workflows
├── README.md         # This file
├── CODEOWNERS        # Code ownership rules
└── pull_request_template.md
```

## Key Differences

### Composite Actions (`.github/actions/`)

- **Purpose**: Reusable steps that can be called within workflows
- **Structure**: Use `runs.using: 'composite'` with `inputs` and `steps`
- **Usage**: Called with `uses: ./.github/actions/action-name`
- **Scope**: Individual steps within a job
- **Examples**: Running tests, uploading artifacts, updating badges

### Reusable Workflows (`.github/workflows/`)

- **Purpose**: Complete workflows that can be called by other workflows
- **Structure**: Use `on.workflow_call` with `inputs`, `jobs`, and `secrets`
- **Usage**: Called with `uses: ./.github/workflows/workflow-name.yml`
- **Scope**: Complete jobs or entire workflows
- **Examples**: Notifications, complex testing scenarios, multi-job processes

## When to Use Each

### Use Composite Actions When

- You need a reusable step within a job
- The functionality is simple and focused
- You want to avoid duplicating setup code (Python setup, checkout, etc.)
- You're building a step that will be used across multiple workflows

### Use Reusable Workflows When

- You need complete workflow functionality with multiple jobs
- The process involves complex logic or multiple steps
- You want to trigger the workflow independently
- You need to handle secrets or environment variables
- You're building notification systems or complex CI/CD processes

## Current Usage Patterns

### Composite Actions in Use

- `run-coverage-tests/` - Runs tests with coverage checking
- `display-coverage-comment/` - Displays coverage results as PR comments
- `upload-coverage-report/` - Uploads coverage reports as artifacts
- `update-badge/` - Updates coverage badge in README

### Reusable Workflows in Use

- `reusable-send-notifications.yml` - Sends Slack and email notifications
- `reusable-send-email-notification.yml` - Dedicated email notifications
- `reusable-run-coverage-tests.yml` - Complete coverage testing workflow
- `reusable-diff-based-test-coverage.yml` - Advanced diff-based testing
- `reusable-ruff-code-linting.yml` - Code linting workflow
- `reusable-taskgraph-generation-validation.yml` - Task graph validation
- `reusable-display-coverage-comment.yml` - Display coverage comments
- `reusable-upload-coverage-report.yml` - Upload coverage reports
- `reusable-update-badge.yml` - Update coverage badges
- `reusable-readme-coverage-badge-update.yml` - Update README with coverage badge

## Best Practices

1. **Consistent Naming**: Use `reusable-` prefix for reusable workflows
2. **Clear Documentation**: Document inputs, outputs, and usage examples
3. **Error Handling**: Include proper error handling and logging
4. **Default Values**: Provide sensible defaults for optional inputs
5. **Secrets Management**: Use `secrets: inherit` when calling reusable workflows
6. **Testing**: Test both actions and workflows before deploying

## Setup Requirements

### Required Secrets

- `SLACK_WEBHOOK_URL`: For Slack notifications
- `SENDGRID_API_KEY`: For email notifications via SendGrid
- `EMAIL_WEBHOOK_URL`: For email notifications via webhook

### Environment Setup

- Python 3.10+ for most workflows
- Coverage tools (pytest, coverage)
- Linting tools (Ruff)

## Documentation

- [Actions Documentation](actions/README.md) - Details about composite actions
- [Workflows Documentation](workflows/README.md) - Details about reusable workflows

## Examples

### Using a Composite Action

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/run-coverage-tests
        with:
          min-coverage-threshold: '99.0'
```

### Using a Reusable Workflow

```yaml
jobs:
  notify:
    uses: ./.github/workflows/reusable-send-notifications.yml
    with:
      notification-type: 'success'
      title: '✅ Build Successful'
      message: 'All tests passed!'
      workflow-name: 'CI Pipeline'
      workflow-run-id: ${{ github.run_id }}
      workflow-run-url: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
    secrets: inherit
```
