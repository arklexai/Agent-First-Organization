# GitHub Workflows - Reusable Workflows

This directory contains reusable GitHub Workflows that can be called by other workflows or triggered by events.

## Available Reusable Workflows

### Notification Workflows

#### 1. `reusable-send-notifications.yml`

Main notification workflow that can send both Slack and email notifications.

**Inputs:**

- `notification-type`: Type of notification (success, failure, warning, info) (default: info)
- `title`: Notification title
- `message`: Notification message content
- `workflow-name`: Name of the workflow that triggered the notification
- `workflow-run-id`: GitHub workflow run ID
- `workflow-run-url`: URL to the workflow run
- `enable-email`: Whether to send email notifications (default: true)
- `enable-slack`: Whether to send Slack notifications (default: true)
- `email-service`: Email service to use (sendgrid, webhook, debug) (default: debug)
- `recipient-email`: Email address to send notification to (default: <christian.lim@arklexai.com>)

#### 2. `reusable-send-email-notification.yml`

Dedicated email notification workflow with multiple service options.

**Inputs:**

- `subject`: Email subject line
- `body`: Email body content (HTML supported)
- `recipient-email`: Email address to send notification to (default: <christian.lim@arklexai.com>)
- `sender-name`: Name of the sender (default: GitHub Actions Bot)
- `sender-email`: Email address of the sender (default: <noreply@arklexai.com>)
- `email-service`: Email service to use (sendgrid, webhook, debug) (default: debug)

### Coverage and Testing Workflows

#### 3. `reusable-run-coverage-tests.yml`

Reusable workflow to run tests with coverage checking.

**Inputs:**

- `coverage-command`: Command to run tests with coverage (default: pytest with coverage)
- `min-coverage-threshold`: Minimum coverage percentage to pass (default: 99.0)

#### 4. `reusable-diff-based-test-coverage.yml`

Advanced coverage workflow that runs tests only for changed files.

**Inputs:**

- `coverage-command`: Command to run tests with coverage
- `min-coverage-threshold`: Minimum coverage percentage to pass
- `diff-base`: Base branch for diff calculation (default: main)

#### 5. `reusable-display-coverage-comment.yml`

Workflow to display coverage results as comments on PRs.

#### 6. `reusable-upload-coverage-report.yml`

Workflow to upload coverage reports as artifacts.

#### 7. `reusable-update-badge.yml`

Workflow to update coverage badge in README.

#### 8. `reusable-readme-coverage-badge-update.yml`

Workflow to update README with coverage badge.

### Task Graph Workflows

#### 9. `reusable-taskgraph-generation-validation.yml`

Workflow for validating task graph generation.

### Code Quality Workflows

#### 10. `reusable-ruff-code-linting.yml`

Workflow for running Ruff code linting.

## Standalone Workflows

### Event-Triggered Workflows

#### 1. `nightly-regression.yml`

Nightly regression testing workflow.

#### 2. `test-coverage.yml`

Test coverage workflow triggered by PR labels.

#### 3. `pr-taskgraph-validation.yml`

PR validation for task graphs.

#### 4. `pr-code-linting.yml`

PR code linting workflow.

#### 5. `pr-diff-coverage-check.yml`

PR diff-based coverage checking.

#### 6. `pr-check.yml`

General PR checking workflow.

#### 7. `release.yml`

Release workflow.

#### 8. `test-resources.yml`

Test resources workflow.

## Usage Examples

### Calling Reusable Workflows

```yaml
# In another workflow file
jobs:
  notify-success:
    uses: ./.github/workflows/reusable-send-notifications.yml
    with:
      notification-type: 'success'
      title: '✅ Build Successful'
      message: 'The build has completed successfully.'
      workflow-name: 'CI Pipeline'
      workflow-run-id: ${{ github.run_id }}
      workflow-run-url: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
    secrets: inherit

  run-coverage:
    uses: ./.github/workflows/reusable-run-coverage-tests.yml
    with:
      min-coverage-threshold: '99.0'
```

## Setup Instructions

### Required Secrets

Add these secrets to your GitHub repository:

- `SLACK_WEBHOOK_URL`: Slack incoming webhook URL (for Slack notifications)
- `SENDGRID_API_KEY`: SendGrid API key (for email notifications via SendGrid)
- `EMAIL_WEBHOOK_URL`: Webhook URL for email service (for email notifications via webhook)

### Email Service Options

1. **SendGrid**: Set `email-service: 'sendgrid'` and provide `SENDGRID_API_KEY`
2. **Webhook**: Set `email-service: 'webhook'` and provide `EMAIL_WEBHOOK_URL`
3. **Debug**: Set `email-service: 'debug'` to print emails to logs (default)

## Differences from Composite Actions

- **Workflows** are complete workflows that can contain multiple jobs
- **Actions** are composite steps that run within a single job
- Workflows can be triggered by events or called by other workflows
- Actions are always called as steps within workflows
- Workflows have more complex structure with jobs, steps, and environment setup
- Actions are more focused and lightweight

## Notes

- Reusable workflows are designed to be called by other workflows using `uses:`
- Event-triggered workflows run automatically based on GitHub events
- All workflows include proper error handling and logging
- Debug mode is enabled by default for email notifications to prevent accidental emails
