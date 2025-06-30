"""Setup configuration for the TAU benchmark package.

This module defines the package configuration for the TAU benchmark, including
package metadata, dependencies, and installation requirements. It uses setuptools
to configure the package for distribution and installation.
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tau_bench",
    version="0.1.0",
    description="The Tau-Bench package",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=1.13.3",
        "mistralai>=0.4.0",
        "anthropic>=0.26.1",
        "google-generativeai>=0.5.4",
        "tenacity>=8.3.0",
        "termcolor>=2.4.0",
        "numpy>=1.26.4",
        "litellm>=1.41.0",
    ],
)
