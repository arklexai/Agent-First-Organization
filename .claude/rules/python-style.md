---
paths:
  - "**/*.py"
---

# Python style rules

- Ruff format + ruff check enforced by hooks (auto-runs on every edit)
- Line length: 88 (configured in pyproject.toml)
- Lint rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade), ANN (annotations), B (bugbear), SIM (simplify), C4 (comprehensions), TID (tidy-imports)
- Ignored: E501 (line too long, handled by formatter)
- Format: double quotes, space indent, no magic trailing comma skip, auto line endings
- Absolute imports only
- Type annotations on all function signatures
- No license header required (MIT license)
- Target version: Python 3.10
