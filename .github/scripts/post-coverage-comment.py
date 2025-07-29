#!/usr/bin/env python3
"""
Custom script to post coverage comments to GitHub PRs.
This script reads coverage.xml and posts a formatted comment.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any

import requests


def parse_coverage_xml(file_path: str) -> dict[str, Any]:
    """Parse coverage.xml and extract coverage information."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract coverage data
        line_rate = float(root.attrib.get("line-rate", 0))
        branch_rate = float(root.attrib.get("branch-rate", 0))
        lines_covered = int(root.attrib.get("lines-covered", 0))
        lines_valid = int(root.attrib.get("lines-valid", 0))

        # Calculate percentage
        coverage_percentage = line_rate * 100

        return {
            "line_rate": line_rate,
            "branch_rate": branch_rate,
            "lines_covered": lines_covered,
            "lines_valid": lines_valid,
            "coverage_percentage": coverage_percentage,
        }
    except Exception as e:
        print(f"Error parsing coverage.xml: {e}")
        return None


def get_coverage_status(
    coverage_percentage: float, min_green: float = 99.0, min_orange: float = 70.0
) -> str:
    """Determine coverage status based on thresholds."""
    if coverage_percentage >= min_green:
        return "🟢"
    elif coverage_percentage >= min_orange:
        return "🟡"
    else:
        return "🔴"


def format_coverage_comment(
    coverage_data: dict[str, Any], min_green: float = 99.0, min_orange: float = 70.0
) -> str:
    """Format coverage data into a GitHub comment."""
    coverage_percentage = coverage_data["coverage_percentage"]
    status = get_coverage_status(coverage_percentage, min_green, min_orange)

    comment = f"""## 📊 Code Coverage Report

{status} **Overall Coverage: {coverage_percentage:.1f}%**

### 📈 Coverage Details:
- **Lines Covered:** {coverage_data["lines_covered"]:,} / {coverage_data["lines_valid"]:,}
- **Line Coverage:** {coverage_data["line_rate"]:.1%}
- **Branch Coverage:** {coverage_data["branch_rate"]:.1%}

### 🎯 Thresholds:
- 🟢 **Green:** ≥ {min_green}%
- 🟡 **Orange:** ≥ {min_orange}%
- 🔴 **Red:** < {min_orange}%

---
*This report was generated automatically by the coverage workflow.*"""

    return comment


def post_github_comment(repo: str, pr_number: int, comment: str, token: str) -> bool:
    """Post comment to GitHub PR."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    data = {"body": comment}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"✅ Comment posted successfully to PR #{pr_number}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to post comment: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def main() -> None:
    """Main function."""
    # Get environment variables
    coverage_path = os.environ.get("COVERAGE_PATH", "coverage.xml")
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repository = os.environ.get("GITHUB_REPOSITORY")
    github_event_path = os.environ.get("GITHUB_EVENT_PATH")
    min_green = float(os.environ.get("MINIMUM_GREEN", "99.0"))
    min_orange = float(os.environ.get("MINIMUM_ORANGE", "70"))

    # Validate required environment variables
    if not github_token:
        print("❌ GITHUB_TOKEN environment variable is required")
        sys.exit(1)

    if not github_repository:
        print("❌ GITHUB_REPOSITORY environment variable is required")
        sys.exit(1)

    if not github_event_path:
        print("❌ GITHUB_EVENT_PATH environment variable is required")
        sys.exit(1)

    # Read GitHub event to get PR number
    try:
        with open(github_event_path) as f:
            event_data = json.load(f)

        pr_number = event_data.get("number")
        if not pr_number:
            print("❌ Could not determine PR number from event data")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading GitHub event: {e}")
        sys.exit(1)

    # Parse coverage data
    print(f"📊 Parsing coverage data from {coverage_path}...")
    coverage_data = parse_coverage_xml(coverage_path)
    if not coverage_data:
        print("❌ Failed to parse coverage data")
        sys.exit(1)

    print(f"✅ Coverage: {coverage_data['coverage_percentage']:.1f}%")

    # Format comment
    comment = format_coverage_comment(coverage_data, min_green, min_orange)

    # Post comment
    print(f"💬 Posting comment to PR #{pr_number}...")
    success = post_github_comment(github_repository, pr_number, comment, github_token)

    if success:
        print("🎉 Coverage comment posted successfully!")
        sys.exit(0)
    else:
        print("❌ Failed to post coverage comment")
        sys.exit(1)


if __name__ == "__main__":
    main()
