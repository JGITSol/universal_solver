"""Tests for project metadata and configuration."""

import re
from pathlib import Path

import pytest


def test_readme_has_badges():
    """Test that README.md contains industry-standard badges."""
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md should exist"

    content = readme_path.read_text(encoding="utf-8")

    # Check for required badge categories
    assert "[![CI]" in content, "CI badge should be present"
    assert "[![Test Coverage]" in content, "Test coverage badge should be present"
    assert "[![pre-commit]" in content, "Pre-commit badge should be present"
    assert "[![Python Version]" in content, "Python version badge should be present"
    assert "[![Code style: black]" in content, "Black badge should be present"
    assert "[![License: MIT]" in content, "License badge should be present"
    assert "[![GitHub stars]" in content, "GitHub stars badge should be present"
    assert "[![GitHub issues]" in content, "GitHub issues badge should be present"


def test_readme_badge_links_valid():
    """Test that badge links in README.md are properly formatted."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    # Find all badge patterns
    badge_pattern = r"\[!\[.*?\]\((.*?)\)\]\((.*?)\)"
    matches = re.findall(badge_pattern, content)

    assert len(matches) > 0, "Should have badge links"

    # Check that URLs are valid (basic check)
    for img_url, link_url in matches:
        assert (
            img_url.startswith("http")
            or img_url.startswith("./")
            or img_url.startswith("https")
        ), f"Badge image URL should be valid: {img_url}"
        assert (
            link_url.startswith("http")
            or link_url.startswith("./")
            or link_url.startswith("https")
        ), f"Badge link URL should be valid: {link_url}"


def test_setup_py_has_classifiers():
    """Test that setup.py contains comprehensive PyPI classifiers."""
    setup_path = Path(__file__).parent.parent / "setup.py"
    content = setup_path.read_text(encoding="utf-8")

    # Check for key classifier categories
    assert "Development Status :: 3 - Alpha" in content
    assert "Intended Audience :: Developers" in content
    assert "Intended Audience :: Science/Research" in content
    assert "Topic :: Scientific/Engineering :: Mathematics" in content
    assert "License :: OSI Approved :: MIT License" in content
    assert "Programming Language :: Python :: 3.10" in content
    assert "Operating System :: OS Independent" in content


def test_setup_py_has_keywords():
    """Test that setup.py contains relevant keywords."""
    setup_path = Path(__file__).parent.parent / "setup.py"
    content = setup_path.read_text(encoding="utf-8")

    # Check for important keywords
    assert "mathematics" in content.lower()
    assert "solver" in content.lower()
    assert "machine-learning" in content.lower()


def test_setup_py_has_project_urls():
    """Test that setup.py includes project URLs."""
    setup_path = Path(__file__).parent.parent / "setup.py"
    content = setup_path.read_text(encoding="utf-8")

    assert "project_urls" in content
    assert "Bug Tracker" in content
    assert "Documentation" in content
    assert "Source Code" in content


def test_license_file_exists():
    """Test that LICENSE file exists."""
    license_path = Path(__file__).parent.parent / "LICENSE"
    assert license_path.exists(), "LICENSE file should exist"

    content = license_path.read_text(encoding="utf-8")
    assert "MIT License" in content, "Should be MIT License"
    assert "Permission is hereby granted" in content


def test_version_file_exists():
    """Test that VERSION file exists and has valid format."""
    version_path = Path(__file__).parent.parent / "VERSION"
    assert version_path.exists(), "VERSION file should exist"

    version = version_path.read_text(encoding="utf-8").strip()
    # Should match semver pattern (basic check)
    assert re.match(r"\d+\.\d+\.\d+", version), "Version should follow semver format"
