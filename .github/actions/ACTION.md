# GitHub Actions - Reusable Actions

This directory contains reusable GitHub Actions (composite actions) that can be used as steps within workflows.

## Available Actions

### 1. `run-coverage-tests/`

Composite action to run tests with coverage and check against minimum threshold.

**Inputs:**

- `coverage-command`: Command to run tests with coverage (default: pytest with coverage)
- `min-coverage-threshold`: Minimum coverage percentage to pass (default: 99.0)
- `checkout-repo`: Whether to checkout the repository (default: true)

### 2. `display-coverage-comment/`

Composite action to display coverage results as a comment on PRs.

**Inputs:**

- `github-token`: GitHub token for authentication (required)
- `minimum-green`: Minimum coverage percentage for green status (default: 99.0)
- `minimum-orange`: Minimum coverage percentage for orange status (default: 70)

### 3. `upload-coverage-report/`

Composite action to upload coverage reports as artifacts.

**Inputs:**

- `coverage-xml-path`: Path to coverage.xml file (default: coverage.xml)
- `htmlcov-path`: Path to htmlcov directory (default: htmlcov/)
- `artifact-name`: Name for the coverage report artifact (default: coverage-report)
- `htmlcov-artifact-name`: Name for the HTML coverage artifact (default: htmlcov)

### 4. `update-badge/`

Composite action to update coverage badge in README.

**Inputs:**

- `coverage-file`: Path to coverage file (coverage.txt or coverage.xml) (default: coverage.txt)
- `coverage-format`: Format of coverage file (txt or xml) (default: txt)
- `min-coverage-threshold`: Minimum coverage threshold for color coding (default: 99.0)
- `badge-pattern`: Pattern to match existing badge in README (default: coverage-[0-9]+\.[0-9]+%25-[a-z]+)

## Usage Examples

### Using Composite Actions in Workflows

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: pip install -e '.[milvus,shopify,hubspot]'
        
      - name: Run coverage tests
        uses: ./.github/actions/run-coverage-tests
        with:
          min-coverage-threshold: '99.0'
          
      - name: Display coverage comment
        uses: ./.github/actions/display-coverage-comment
        with:
          github-token: ${{ github.token }}
          minimum-green: '99.0'
          minimum-orange: '70'
```

## Differences from Reusable Workflows

- **Actions** are composite steps that run within a single job
- **Workflows** are complete workflows that can be called by other workflows
- Actions are more lightweight and focused on specific tasks
- Workflows can contain multiple jobs and more complex logic

## Notes

- These actions are designed to be reusable across different workflows
- They handle common CI/CD tasks like testing, coverage reporting, and badge updates
- For notification functionality, see the reusable workflows in `.github/workflows/`
