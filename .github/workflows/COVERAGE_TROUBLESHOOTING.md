# Coverage Workflow Troubleshooting Guide

## Common Issues and Solutions

### 1. 406 Not Acceptable Error

**Error Message:**

```
Error: Critical error. This error possibly occurred because the permissions of the workflow are set incorrectly.
HTTPStatusError: Client error '406 Not Acceptable' for url 'https://api.github.com/repos/.../pulls/266'
```

**Causes:**

- Insufficient GitHub token permissions
- PR doesn't exist or is not accessible
- Workflow permissions not properly configured

**Solutions:**

1. **Check Workflow Permissions:**

   ```yaml
   permissions:
     pull-requests: write
     contents: write
     actions: read
     checks: write
   ```

2. **Use Correct GitHub Token:**
   - Use `${{ secrets.GITHUB_TOKEN }}` instead of `${{ github.token }}`
   - Ensure the token has sufficient permissions

3. **Verify PR Exists:**
   - Check if the PR number is correct
   - Ensure the PR is accessible to the workflow

### 2. Coverage File Issues

**Error Message:**

```
Error: coverage.xml file not found
```

**Solutions:**

1. **Verify pytest command:**

   ```bash
   pytest tests/ --cov=arklex --cov-report=term-missing --cov-report=html --cov-report=xml
   ```

2. **Check coverage.xml format:**

   ```bash
   python .github/scripts/validate-coverage.py coverage.xml
   ```

3. **Ensure coverage package is installed:**

   ```bash
   pip install pytest-cov
   ```

### 3. GitHub API Access Issues

**Error Message:**

```
Error: Cannot access PR information
```

**Solutions:**

1. **Test API access manually:**

   ```bash
   curl -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/owner/repo/pulls/PR_NUMBER"
   ```

2. **Check repository permissions:**
   - Ensure the workflow has access to the repository
   - Verify the GitHub token has the correct scope

### 4. Workflow Trigger Issues

**Problem:** Workflow not triggering on PR

**Solutions:**

1. **Check trigger conditions:**

   ```yaml
   on:
     pull_request:
       types: [labeled, opened, synchronize]
   ```

2. **Verify label requirements:**
   - Ensure the PR has the required label (`run-coverage-tests`)
   - Or use manual trigger with `workflow_dispatch`

## Debugging Steps

### 1. Run Test Access Workflow

Use the `test-github-access.yml` workflow to verify GitHub API access:

```bash
# Trigger manually from GitHub Actions tab
# Or add the label to a PR
```

### 2. Check Coverage File

```bash
# Validate coverage.xml format
python .github/scripts/validate-coverage.py coverage.xml

# Check file contents
head -20 coverage.xml
```

### 3. Test Coverage Action Locally

```bash
# Install the action locally for testing
npm install -g @actions/core
```

## Fallback Mechanisms

The workflow includes several fallback mechanisms:

1. **Manual Coverage Comment:** If the py-cov-action fails, a manual comment is posted
2. **Error Handling:** The workflow continues even if the coverage comment fails
3. **Validation Steps:** Multiple validation steps ensure data integrity

## Best Practices

1. **Always use `secrets.GITHUB_TOKEN`** instead of `github.token`
2. **Include comprehensive permissions** in workflow configuration
3. **Add validation steps** before using external actions
4. **Implement fallback mechanisms** for critical functionality
5. **Test workflows manually** before relying on automated triggers

## Support

If issues persist:

1. Check the GitHub Actions logs for detailed error messages
2. Verify repository settings and permissions
3. Test with a simple PR to isolate the issue
4. Consider using the manual trigger (`workflow_dispatch`) for testing
