#!/usr/bin/env python3
"""
Script to validate coverage.xml file format and content.
"""

import os
import sys
import xml.etree.ElementTree as ET


def validate_coverage_xml(file_path: str) -> bool | None:
    """Validate the coverage.xml file format and content."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} does not exist")
            return False

        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Check root element
        if root.tag != "coverage":
            print(f"❌ Error: Root element should be 'coverage', found '{root.tag}'")
            return False

        # Check required attributes
        required_attrs = [
            "line-rate",
            "branch-rate",
            "lines-covered",
            "lines-valid",
            "branches-covered",
            "branches-valid",
        ]
        for attr in required_attrs:
            if attr not in root.attrib:
                print(f"❌ Error: Missing required attribute '{attr}'")
                return False

        # Check if there are any packages
        packages = root.findall(".//package")
        if not packages:
            print("❌ Error: No packages found in coverage data")
            return False

        # Print coverage summary
        line_rate = float(root.attrib["line-rate"]) * 100
        branch_rate = float(root.attrib["branch-rate"]) * 100
        lines_covered = int(root.attrib["lines-covered"])
        lines_valid = int(root.attrib["lines-valid"])

        print("✅ Coverage file is valid")
        print(f"📊 Line coverage: {line_rate:.1f}% ({lines_covered}/{lines_valid})")
        print(f"📊 Branch coverage: {branch_rate:.1f}%")
        print(f"📦 Packages found: {len(packages)}")

        return True

    except ET.ParseError as e:
        print(f"❌ Error: Invalid XML format - {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def validate_coverage_database() -> bool | None:
    """Validate the .coverage database file."""
    try:
        if not os.path.exists(".coverage"):
            print("❌ Error: .coverage database file does not exist")
            return False

        # Check file size (should be non-zero)
        file_size = os.path.getsize(".coverage")
        if file_size == 0:
            print("❌ Error: .coverage database file is empty")
            return False

        print(f"✅ .coverage database file is valid (size: {file_size} bytes)")
        return True

    except Exception as e:
        print(f"❌ Error validating .coverage database: {e}")
        return False


def main() -> int:
    """Main validation function."""
    print("🔍 Validating coverage files...\n")

    # Validate .coverage database
    print("1. Checking .coverage database file:")
    db_valid = validate_coverage_database()

    # Validate coverage.xml
    print("\n2. Checking coverage.xml file:")
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "coverage.xml"
    xml_valid = validate_coverage_xml(xml_file)

    # Summary
    print("\n📊 Validation Summary:")
    print(f"   .coverage database: {'✅ Valid' if db_valid else '❌ Invalid'}")
    print(f"   coverage.xml: {'✅ Valid' if xml_valid else '❌ Invalid'}")

    if db_valid and xml_valid:
        print("\n🎉 All coverage files are valid!")
        return 0
    else:
        print("\n⚠️ Some coverage files have issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
