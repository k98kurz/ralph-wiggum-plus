#!/usr/bin/env python3
"""Test suite for ralph.py prompt generation system."""

import os
import shutil
import sys
import unittest
from pathlib import Path
from string import Template

# Import the main module
import ralph


class TestTemplateGeneration(unittest.TestCase):
    """Test cases for the prompt generation template system."""

    def setUp(self):
        """Set up test environment before each test."""
        # Save original working directory
        self.original_cwd = os.getcwd()
        
        # Create a test directory
        self.test_dir = Path("test_ralph_temp")
        self.test_dir.mkdir(exist_ok=True)
        os.chdir(self.test_dir)
        
        # Initialize RWLState with minimal required fields
        self.state = ralph.RWLState(
            original_prompt="Test prompt for template generation",
            iteration=1,
            current_phase=ralph.Phase.BUILD.value,
            max_iterations=10,
            model="test/model",
            review_model="test/review-model",
        )
        
        # Track created artifacts for cleanup
        self.artifacts = []

    def tearDown(self):
        """Clean up test environment after each test."""
        # Change back to original directory
        os.chdir(self.original_cwd)
        
        # Remove test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Clear global template cache
        ralph.PROMPT_TEMPLATES.clear()

    def test_template_generation_with_defaults(self):
        """Test that templates generate correctly with default values (no custom templates)."""
        result = ralph.check_template_generation(self.state), (self.state)
        
        assert result[0] == True, (
            "Template generation should succeed with default templates",
            f"State: {self.state}, Result: {result}"
        )

    def test_template_generation_with_malformed_custom_template(self):
        """Test that malformed custom templates return False."""
        # Create .ralph/templates directory
        template_dir = Path(".ralph/templates")
        template_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.append(template_dir)
        
        # Create a malformed custom template with invalid variable
        malformed_template = """
        You are a software engineer.
        
        Task: $invalid_variable_name
        """
        
        build_template_path = template_dir / "build.md"
        build_template_path.write_text(malformed_template)
        self.artifacts.append(build_template_path)
        
        # Clear template cache to force reading from disk
        ralph.PROMPT_TEMPLATES.clear()
        
        result = ralph.check_template_generation(self.state), (self.state)
        
        assert result[0] == False, (
            "Template generation should fail with malformed custom template",
            f"State: {self.state}, Result: {result}, Template: {malformed_template}"
        )


if __name__ == "__main__":
    unittest.main()
