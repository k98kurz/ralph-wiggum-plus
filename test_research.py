#!/usr/bin/env python3
"""Test suite for research.py prompt generation system."""

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

import research


class TestWithTempDir(unittest.TestCase):
    """Base test class with temporary directory setup/teardown."""

    def setUp(self):
        """Set up temporary test directory."""
        self.original_cwd = os.getcwd()
        self.test_dir = Path(f"test_{secrets.token_hex(8)}")
        self.test_dir.mkdir(exist_ok=True)
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up temporary test directory."""
        os.chdir(self.original_cwd)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)


class TestTemplateGeneration(TestWithTempDir):
    """Test cases for the prompt generation template system."""

    def setUp(self):
        """Set up test environment before each test."""
        super().setUp()

        self.state = research.RWRState(
            original_topic="Test research topic",
            breadth=3,
            depth=3,
            iteration=1,
            current_phase=research.Phase.RESEARCH.value,
            max_iterations=20,
            model="test/model",
            review_model="test/review-model",
            research_name="test-research",
        )

    def tearDown(self):
        """Clean up test environment after each test."""
        research.PROMPT_TEMPLATES.clear()
        super().tearDown()

    def test_template_generation_with_defaults(self):
        """Test that templates generate correctly with default values (no custom templates)."""
        result = research.check_template_generation(self.state)

        assert result == True, (
            "Template generation should succeed with default templates",
            f"State: {self.state}, Result: {result}"
        )

    def test_template_generation_with_malformed_custom_template(self):
        """Test that malformed custom templates return False."""
        template_dir = Path(f".research/{self.state.research_name}/templates")
        template_dir.mkdir(parents=True, exist_ok=True)

        malformed_template = """
        You are a research assistant.

        Task: $invalid_variable_name
        """

        research_template_path = template_dir / "research.md"
        research_template_path.write_text(malformed_template)

        research.PROMPT_TEMPLATES.clear()

        result = research.check_template_generation(self.state)

        assert result == False, (
            "Template generation should fail with malformed custom template",
            f"State: {self.state}, Result: {result}, Template: {malformed_template}"
        )


class TestArchiveSystem(TestWithTempDir):
    """Test cases for the archiving system."""

    def setUp(self):
        """Set up environment before each test."""
        super().setUp()
        subprocess.run(["git", "init"], capture_output=True, check=True)

        self.research_name = f"test-research-{secrets.token_hex(8)}"
        self.lock_token = secrets.token_hex(16)
        research.ARCHIVED_HASHES.clear()

        research_dir = Path(f".research/{self.research_name}")
        research_dir.mkdir(parents=True, exist_ok=True)

        archive_dir = research_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        self.state = research.RWRState(
            research_name=self.research_name,
            lock_token=self.lock_token,
            original_topic="Test topic"
        )

    def tearDown(self):
        """Clean up after each test."""
        research.ARCHIVED_HASHES.clear()
        super().tearDown()

    def test_archive_intermediate_file_basic(self):
        """Test basic single file archiving."""
        research_dir = Path(f".research/{self.research_name}")
        test_file = research_dir / "progress.md"
        test_file.write_text("Test content")

        archive_dir = research_dir / "archive"
        archives = list(archive_dir.glob(f"*.{self.lock_token}.progress.md"))
        assert len(archives) == 0

        research.archive_intermediate_file(test_file, self.state)
        archives = list(archive_dir.glob(f"*.{self.lock_token}.progress.md"))

        assert len(archives) == 1, (
            "Should create exactly one archive file",
            f"Archives found: {archives}"
        )

        archive_file = archives[0]
        assert archive_file.exists(), (
            "Archive file should exist",
            f"Archive file: {archive_file}"
        )

        assert research.ARCHIVE_FILE_FORMAT.match(archive_file.name), (
            "Archive filename should match expected format",
            f"Filename: {archive_file.name}"
        )

        assert archive_file.read_text() == "Test content", (
            "Archive content should match original file",
            f"Original: 'Test content', Archive: '{archive_file.read_text()}'"
        )

        expected_hash = hashlib.sha256(b"Test content").hexdigest()
        assert expected_hash in research.ARCHIVED_HASHES, (
            "File hash should be added to ARCHIVED_HASHES",
            f"Hash: {expected_hash}, ARCHIVED_HASHES: {research.ARCHIVED_HASHES}"
        )

    def test_archive_intermediate_file_with_prepend(self):
        """Test archiving file with prepend parameter for subtopics."""
        research_dir = Path(f".research/{self.research_name}")
        test_file = research_dir / "progress.md"
        test_file.write_text("Test content for subtopic")

        archive_dir = research_dir / "archive"
        archives = list(archive_dir.glob(f"*.{self.lock_token}.machine-learning.progress.md"))
        assert len(archives) == 0

        research.archive_intermediate_file(test_file, self.state, prepend="machine-learning")
        archives = list(archive_dir.glob(f"*.{self.lock_token}.machine-learning.progress.md"))

        assert len(archives) == 1, (
            "Should create archive with prepend in filename",
            f"Archives found: {archives}"
        )

    def test_archive_intermediate_file_deduplication(self):
        """Test hash-based deduplication prevents duplicate archives."""
        research_dir = Path(f".research/{self.research_name}")
        test_file = research_dir / "progress.md"
        test_file.write_text("Test content")
        archives= list((research_dir / "archive").glob(f"*.{self.lock_token}.progress.md"))
        assert len(archives) == 0

        research.archive_intermediate_file(test_file, self.state)
        archives = list((research_dir / "archive").glob(f"*.{self.lock_token}.progress.md"))

        assert len(archives) == 1, (
            "First archive should create one file",
            f"Archives: {archives}"
        )

        research.archive_intermediate_file(test_file, self.state)
        archives = list((research_dir / "archive").glob(f"*.{self.lock_token}.progress.md"))

        assert len(archives) == 1, (
            "Second archive with same content should not create duplicate",
            f"Archives: {archives}"
        )

        time.sleep(1.5)

        test_file.write_text("Modified content")

        research.archive_intermediate_file(test_file, self.state)
        archives = list((research_dir / "archive").glob(f"*.{self.lock_token}.progress.md"))

        assert len(archives) == 2, (
            "Modified content should create new archive",
            f"Archives: {archives}"
        )

    def test_archive_intermediate_file_nonexistent(self):
        """Test graceful handling of missing files."""
        research_dir = Path(f".research/{self.research_name}")
        nonexistent_file = research_dir / "does_not_exist.md"

        research.archive_intermediate_file(nonexistent_file, self.state)

        archive_dir = research_dir / "archive"
        archives = list(archive_dir.glob(f"*.{self.lock_token}.does_not_exist.md"))

        assert len(archives) == 0, (
            "Should not create archive for nonexistent file",
            f"Archives: {archives}"
        )

        assert len(research.ARCHIVED_HASHES) == 0, (
            "ARCHIVED_HASHES should remain empty",
            f"ARCHIVED_HASHES: {research.ARCHIVED_HASHES}"
        )

    def test_archive_any_process_files(self):
        """Test batch archiving of all tracked process files."""
        research_dir = Path(f".research/{self.research_name}")
        progress_file = research_dir / "progress.md"
        progress_file.write_text("Progress content")

        accepted_file = research_dir / "review.accepted.md"
        accepted_file.write_text("Review accepted")

        rejected_file = research_dir / "review.rejected.md"
        rejected_file.write_text("Review rejected")

        control_file = research_dir / "ignored_file.txt"
        control_file.write_text("Ignored content")

        archive_dir = research_dir / "archive"
        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 0, (
            "Archive directory should be empty before test",
            f"Found: {archives_before}"
        )

        research.ARCHIVED_HASHES.clear()

        research.archive_any_process_files(self.state)

        archive_dir = research_dir / "archive"

        archived_progress = list(archive_dir.glob(f"*.{self.lock_token}.progress.md"))
        archived_accepted = list(archive_dir.glob(f"*.{self.lock_token}.review.accepted.md"))
        archived_rejected = list(archive_dir.glob(f"*.{self.lock_token}.review.rejected.md"))

        assert len(archived_progress) == 1, (
            "progress.md should be archived",
            f"Archived progress files: {archived_progress}"
        )

        assert len(archived_accepted) == 1, (
            "review.accepted.md should be archived",
            f"Archived accepted files: {archived_accepted}"
        )

        assert len(archived_rejected) == 1, (
            "review.rejected.md should be archived",
            f"Archived rejected files: {archived_rejected}"
        )

        archived_control = list(archive_dir.glob(f"*.{self.lock_token}.ignored_file.txt"))
        assert len(archived_control) == 0, (
            "Ignored file should not be archived",
            f"Archived ignored files: {archived_control}"
        )

    def test_archive_progress_subdirectory_files(self):
        """Test archiving progress files from subdirectories."""
        research_dir = Path(f".research/{self.research_name}")
        progress_dir = research_dir / "progress"
        subtopic_dir = progress_dir / "machine-learning"
        subtopic_dir.mkdir(parents=True, exist_ok=True)

        progress_file = subtopic_dir / "findings.md"
        progress_file.write_text("ML findings content")

        archive_dir = research_dir / "archive"
        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 0, (
            "Archive directory should be empty before test",
            f"Found: {archives_before}"
        )

        research.ARCHIVED_HASHES.clear()

        research.archive_any_process_files(self.state)

        archive_dir = research_dir / "archive"
        archived_ml = list(archive_dir.glob(f"*.{self.lock_token}.machine-learning.findings.md"))

        assert len(archived_ml) == 1, (
            "Progress subdirectory files should be archived with prepend",
            f"Archived ML files: {archived_ml}"
        )

    def test_populate_archived_hashes(self):
        """Test hash loading from existing archives."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"

        for i in range(3):
            archive_name = f"1234567890.{self.lock_token}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        different_state = research.RWRState(
            research_name=self.research_name,
            lock_token=secrets.token_hex(16)
        )
        different_archive_name = f"1234567890.{different_state.lock_token}.other.md"
        different_archive_path = archive_dir / different_archive_name
        different_archive_path.write_text("Other content")

        research.ARCHIVED_HASHES.clear()

        research.populate_archived_hashes(self.state)

        assert len(research.ARCHIVED_HASHES) == 3, (
            "Should load exactly 3 hashes for matching token",
            f"ARCHIVED_HASHES count: {len(research.ARCHIVED_HASHES)}"
        )

        for i in range(3):
            expected_hash = hashlib.sha256(f"Content {i}".encode()).hexdigest()
            assert expected_hash in research.ARCHIVED_HASHES, (
                f"Hash for file{i} should be loaded",
                f"Expected hash: {expected_hash}"
            )

        other_hash = hashlib.sha256(b"Other content").hexdigest()
        assert other_hash not in research.ARCHIVED_HASHES, (
            "Different token's hash should not be loaded",
            f"Other hash: {other_hash}"
        )

    def test_reorganize_archive_files_basic_with_git(self):
        """Test reorganization using git mv when git available."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"
        token_A = secrets.token_hex(16)

        for i in range(3):
            archive_name = f"12345678{i}.{token_A}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 3, (
            "Should have 3 archives created before reorganization",
            f"Found: {archives_before}"
        )

        subprocess.run(["git", "add", "."], capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "test commit"],
            capture_output=True,
            check=True
        )

        state = research.RWRState(research_name=self.research_name)
        token_B = secrets.token_hex(16)
        research.reorganize_archive_files(state, token_B)

        session_dir = archive_dir / token_A
        assert session_dir.exists(), (
            f"Session directory {token_A} should be created",
            f"Archive dir contents: {list(archive_dir.iterdir())}"
        )

        archived_files = list(session_dir.glob("*.md"))
        assert len(archived_files) == 3, (
            f"All 3 archives should be moved to {token_A}/",
            f"Archived files: {archived_files}"
        )

        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True
        )
        assert token_A in result.stdout, (
            "Moved files should be tracked in git",
            f"Git status: {result.stdout}"
        )

        token_A_files_in_parent = list(archive_dir.glob(f"*.{token_A}.*"))
        assert len(token_A_files_in_parent) == 0, (
            "Token_A files should not remain in parent directory",
            f"Files in parent: {token_A_files_in_parent}"
        )

    def test_reorganize_archive_files_fallback_to_copy(self):
        """Test fallback to shutil.copy2 when git mv fails."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"
        token_A = secrets.token_hex(16)

        for i in range(3):
            archive_name = f"12345678{i}.{token_A}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 3, (
            "Should have 3 archives created before reorganization",
            f"Found: {archives_before}"
        )

        state = research.RWRState(research_name=self.research_name)
        token_B = secrets.token_hex(16)
        research.reorganize_archive_files(state, token_B)

        session_dir = archive_dir / token_A
        assert session_dir.exists(), (
            f"Session directory {token_A} should be created via copy2 fallback",
            f"Archive dir contents: {list(archive_dir.iterdir())}"
        )

        archived_files = list(session_dir.glob("*.md"))
        assert len(archived_files) == 3, (
            f"All 3 archives should be moved to {token_A}/",
            f"Archived files: {archived_files}"
        )

        token_A_files_in_parent = list(archive_dir.glob(f"*.{token_A}.*"))
        assert len(token_A_files_in_parent) == 0, (
            "Original files should be deleted after copy2",
            f"Files in parent: {token_A_files_in_parent}"
        )

    def test_reorganize_archive_files_skips_single_files(self):
        """Test that single-file sessions are NOT reorganized."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"
        token_A = secrets.token_hex(16)
        token_B = secrets.token_hex(16)

        single_archive_name = f"1234567890.{token_A}.single.md"
        single_archive_path = archive_dir / single_archive_name
        single_archive_path.write_text("Single file content")

        for i in range(3):
            archive_name = f"12345678{i}.{token_B}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 4, (
            "Should have 4 archives (1 single + 3 multi) before reorganization",
            f"Found: {archives_before}"
        )

        state = research.RWRState(research_name=self.research_name)
        token_C = secrets.token_hex(16)
        research.reorganize_archive_files(state, token_C)

        token_A_session_dir = archive_dir / token_A
        assert not token_A_session_dir.exists(), (
            "Single-file session should NOT be reorganized",
            f"Session dir created: {token_A_session_dir}"
        )

        token_A_files_in_parent = list(archive_dir.glob(f"*.{token_A}.*"))
        assert len(token_A_files_in_parent) == 1, (
            "Single file should remain in parent directory",
            f"Token_A files in parent: {token_A_files_in_parent}"
        )

        token_B_session_dir = archive_dir / token_B
        assert token_B_session_dir.exists(), (
            "Multi-file session should be reorganized",
            f"Session not found: {token_B_session_dir}"
        )

        token_B_files_in_parent = list(archive_dir.glob(f"*.{token_B}.*"))
        assert len(token_B_files_in_parent) == 0, (
            "Token_B files should be moved from parent directory",
            f"Token_B files in parent: {token_B_files_in_parent}"
        )

    def test_reorganize_archive_files_skips_active_session(self):
        """Test that active session files are NOT reorganized."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"
        active_token = secrets.token_hex(16)

        for i in range(3):
            archive_name = f"12345678{i}.{active_token}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 3, (
            "Should have 3 archives created before reorganization",
            f"Found: {archives_before}"
        )

        state = research.RWRState(research_name=self.research_name)
        research.reorganize_archive_files(state, active_token)

        active_session_dir = archive_dir / active_token
        assert not active_session_dir.exists(), (
            "Active session files should NOT be reorganized",
            f"Session dir created: {active_session_dir}"
        )

        active_token_files = list(archive_dir.glob(f"*.{active_token}.*"))
        assert len(active_token_files) == 3, (
            "Active session files should remain in parent directory",
            f"Active token files: {active_token_files}"
        )

    def test_reorganize_archive_files_duplicate_session_dirs(self):
        """Test handling of existing session directories."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"
        token_A = secrets.token_hex(16)

        existing_session_dir = archive_dir / token_A
        existing_session_dir.mkdir(exist_ok=True)
        (existing_session_dir / "existing.md").write_text("Existing file")

        for i in range(2):
            archive_name = f"12345678{i}.{token_A}.newfile{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"New content {i}")

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 2, (
            "Should have 2 archives created before reorganization",
            f"Found: {archives_before}"
        )

        state = research.RWRState(research_name=self.research_name)
        token_B = secrets.token_hex(16)
        research.reorganize_archive_files(state, token_B)

        new_session_dir = archive_dir / f"{token_A}_1"
        assert new_session_dir.exists(), (
            "Should create {token_A}_1/ to avoid conflict",
            f"Archive dir contents: {list(archive_dir.iterdir())}"
        )

        new_session_files = list(new_session_dir.glob("*.md"))
        assert len(new_session_files) == 2, (
            "New archives should be moved to {token_A}_1/",
            f"New session files: {new_session_files}"
        )

    def test_reorganize_archive_files_no_archives(self):
        """Test graceful handling when no archives exist."""
        archive_dir = Path(f".research/{self.research_name}/archive")

        if archive_dir.exists():
            shutil.rmtree(archive_dir)

        state = research.RWRState(research_name=self.research_name)
        token_A = secrets.token_hex(16)
        research.reorganize_archive_files(state, token_A)

        subdirs = [d for d in archive_dir.iterdir()] if archive_dir.exists() else []
        assert len(subdirs) == 0, (
            "Should not create any directories when no archives exist",
            f"Subdirectories: {subdirs}"
        )

    def test_archive_timestamp_ordering(self):
        """Test that timestamps are monotonically increasing."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 0, (
            "Archive directory should be empty before test",
            f"Found: {archives_before}"
        )

        research.ARCHIVED_HASHES.clear()

        test_file1 = research_dir / "file1.md"
        test_file1.write_text("Content 1")

        test_file2 = research_dir / "file2.md"
        test_file2.write_text("Content 2")

        research.archive_intermediate_file(test_file1, self.state)
        archives_before_sleep = list(archive_dir.glob(f"*.{self.lock_token}.file1.md"))
        timestamp1 = int(archives_before_sleep[0].name.split('.')[0])

        time.sleep(1.5)

        research.archive_intermediate_file(test_file2, self.state)
        archives_after_sleep = list(archive_dir.glob(f"*.{self.lock_token}.file2.md"))
        timestamp2 = int(archives_after_sleep[0].name.split('.')[0])

        assert timestamp1 < timestamp2, (
            "Timestamps should be monotonically increasing",
            f"timestamp1: {timestamp1}, timestamp2: {timestamp2}, difference: {timestamp2 - timestamp1}"
        )

        assert archives_before_sleep[0].name.split('.')[1] == archives_after_sleep[-1].name.split('.')[1], (
            "Both archives should have the same lock_token",
            f"Token1: {archives_before_sleep[0].name.split('.')[1]}, Token2: {archives_after_sleep[-1].name.split('.')[1]}"
        )

    def test_archive_format_regex_validation(self):
        """Test that all generated archives match expected format."""
        research_dir = Path(f".research/{self.research_name}")
        archive_dir = research_dir / "archive"

        test_files = ["file1.md", "file2.txt", "file3.log"]
        test_files = [research_dir / f for f in test_files]

        archives_before = list(archive_dir.glob("*.md"))
        assert len(archives_before) == 0, (
            "Archive directory should be empty before test",
            f"Found: {archives_before}"
        )

        research.ARCHIVED_HASHES.clear()

        for i, test_file in enumerate(test_files):
            test_file.write_text(f"Content {i}")
            research.archive_intermediate_file(test_file, self.state)

        archive_dir = Path(f".research/{self.research_name}/archive")
        archives = list(archive_dir.glob(f"*.{self.lock_token}.*"))

        for archive in archives:
            match = research.ARCHIVE_FILE_FORMAT.match(archive.name)
            assert match is not None, (
                f"Archive filename should match expected format",
                f"Filename: {archive.name}, Pattern: {research.ARCHIVE_FILE_FORMAT.pattern}"
            )

            timestamp_str, lock_token, original_filename = match.groups()

            assert len(lock_token) == 32, (
                f"Lock token should be exactly 32 hex characters",
                f"Lock token length: {len(lock_token)}, value: {lock_token}"
            )

            assert all(c in "0123456789abcdef" for c in lock_token), (
                f"Lock token should be valid hex string",
                f"Lock token: {lock_token}"
            )

            try:
                timestamp = int(timestamp_str)
                assert timestamp > 0, (
                    f"Timestamp should be positive integer",
                    f"Timestamp: {timestamp}"
                )
            except ValueError:
                assert False, (
                    f"Timestamp should be valid integer",
                    f"Timestamp string: {timestamp_str}"
                )

            assert original_filename in ["file1.md", "file2.txt", "file3.log"], (
                f"Original filename should match test files",
                f"Original filename: {original_filename}"
            )


class TestIterationCalculation(unittest.TestCase):
    """Test cases for iteration limit calculation."""

    def test_calculate_min_iterations_default(self):
        """Test default breadth=3, depth=3 calculation."""
        result = research.calculate_min_iterations(3, 3)
        expected = 3 ** (3 + 1) + 5
        assert result == expected, (
            f"Expected {expected}, got {result}"
        )
        assert result == 86, (
            f"Expected 86, got {result}"
        )

    def test_calculate_min_iterations_small(self):
        """Test small breadth=2, depth=2 calculation."""
        result = research.calculate_min_iterations(2, 2)
        expected = 2 ** (2 + 1) + 5
        assert result == expected, (
            f"Expected {expected}, got {result}"
        )
        assert result == 13, (
            f"Expected 13, got {result}"
        )

    def test_calculate_min_iterations_medium(self):
        """Test medium breadth=4, depth=2 calculation."""
        result = research.calculate_min_iterations(4, 2)
        expected = 4 ** (2 + 1) + 5
        assert result == expected, (
            f"Expected {expected}, got {result}"
        )
        assert result == 69, (
            f"Expected 69, got {result}"
        )

    def test_calculate_min_iterations_zero_depth(self):
        """Test depth=0 (just root topics)."""
        result = research.calculate_min_iterations(3, 0)
        expected = 3 ** (0 + 1) + 5
        assert result == 8, (
            f"Expected 8, got {result}"
        )


class TestSlugify(unittest.TestCase):
    """Test cases for slugify function."""

    def test_slugify_basic(self):
        """Test basic slugify functionality."""
        result = research.slugify("Hello World")
        assert result == "hello-world", f"Expected 'hello-world', got '{result}'"

    def test_slugify_special_chars(self):
        """Test slugify with special characters removed."""
        result = research.slugify("Hello, World! Test")
        assert result == "hello-world-test", f"Expected 'hello-world-test', got '{result}'"

    def test_slugify_lowercase(self):
        """Test that slugify converts to lowercase."""
        result = research.slugify("UPPERCASE")
        assert result == "uppercase", f"Expected 'uppercase', got '{result}'"

    def test_slugify_multiple_spaces(self):
        """Test slugify with multiple spaces."""
        result = research.slugify("Multiple   Spaces")
        assert result == "multiple-spaces", f"Expected 'multiple-spaces', got '{result}'"


class TestDirectoryStructure(TestWithTempDir):
    """Test cases for research directory structure creation."""

    def setUp(self):
        super().setUp()

    def test_create_research_directory_structure(self):
        """Test directory structure creation."""
        research_name = "test-session"
        topic = "Test research topic"
        breadth = 3
        depth = 2

        research.create_research_directory_structure(research_name, topic, breadth, depth)

        research_dir = Path(f".research/{research_name}")
        assert research_dir.exists(), "Research directory should exist"

        assert (research_dir / "archive").exists(), "Archive directory should exist"
        assert (research_dir / "progress").exists(), "Progress directory should exist"
        assert (research_dir / "templates").exists(), "Templates directory should exist"
        assert (research_dir / "logs").exists(), "Logs directory should exist"

        reports_dir = Path(f"reports/{research_name}")
        assert reports_dir.exists(), "Reports directory should exist"

        lock_file = research_dir / "research.lock.json"
        assert lock_file.exists(), "Lock file should exist"

        state_file = research_dir / "state.json"
        assert state_file.exists(), "State file should exist"

        progress_file = research_dir / "progress.md"
        assert progress_file.exists(), "Progress file should exist"

        progress_readme = research_dir / "progress" / "README.md"
        assert progress_readme.exists(), "Progress README should exist"


class TestLockingMechanism(TestWithTempDir):
    """Test cases for the locking mechanism."""

    def setUp(self):
        super().setUp()

        subprocess.run(["git", "init"], capture_output=True, check=True)

        self.research_name = "test-research"

        research_dir = Path(f".research/{self.research_name}")
        research_dir.mkdir(parents=True, exist_ok=True)

    def test_get_project_lock_creates_lock(self):
        """Test that get_project_lock creates a lock file."""
        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert not lock_file.exists(), "Lock file should not exist before test"

        token = research.get_project_lock(self.research_name)

        assert token is not None, "Should return a lock token"
        assert len(token) == 32, f"Token should be 32 characters, got {len(token)}"

        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert lock_file.exists(), "Lock file should exist"

        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        assert lock_data["lock_token"] == token, "Lock token should match"
        assert "created" in lock_data, "Lock should have created timestamp"
        assert "pid" in lock_data, "Lock should have pid"

        state = research.RWRState(research_name=self.research_name, lock_token=token)
        assert research.check_project_lock(state)

    def test_get_project_lock_rejects_existing_valid(self):
        """Test that get_project_lock rejects existing valid lock."""
        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert not lock_file.exists(), "Lock file should not exist before test"

        token1 = research.get_project_lock(self.research_name)
        assert token1 is not None

        token2 = research.get_project_lock(self.research_name)
        assert token2 is None, "Should reject existing valid lock"

    def test_get_project_lock_removes_stale_lock(self):
        """Test that stale locks are removed."""
        import time

        research_dir = Path(f".research/{self.research_name}")
        research_dir.mkdir(parents=True, exist_ok=True)

        lock_file = research_dir / "research.lock.json"
        assert not lock_file.exists(), "Lock file should not exist before test"

        stale_lock = {
            "lock_token": "old_token",
            "created": time.time() - 7200,
            "pid": 99999
        }
        with open(lock_file, "w") as f:
            json.dump(stale_lock, f)

        token = research.get_project_lock(self.research_name)
        assert token is not None, "Should get new token after removing stale lock"

    def test_check_project_lock_valid(self):
        """Test check_project_lock with valid token."""
        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert not lock_file.exists(), "Lock file should not exist before test"

        token = research.get_project_lock(self.research_name)
        assert token is not None
        assert lock_file.exists(), "Lock file should exist after creating lock"

        state = research.RWRState(research_name=self.research_name, lock_token=token)
        valid = research.check_project_lock(state)
        assert valid, "Should return True for valid token"

    def test_check_project_lock_invalid(self):
        """Test check_project_lock with invalid token."""
        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert not lock_file.exists(), "Lock file should not exist before test"

        token = research.get_project_lock(self.research_name)
        assert token is not None
        assert lock_file.exists(), "Lock file should exist after creating lock"

        state = research.RWRState(research_name=self.research_name, lock_token="invalid_token")
        invalid = research.check_project_lock(state)
        assert not invalid, "Should return False for invalid token"

    def test_release_project_lock(self):
        """Test release_project_lock removes the lock."""
        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert not lock_file.exists(), "Lock file should not exist before test"

        token = research.get_project_lock(self.research_name)
        assert token is not None
        assert lock_file.exists(), "Lock file should exist after creating lock"

        state = research.RWRState(research_name=self.research_name, lock_token=token)
        research.release_project_lock(state)

        lock_file = Path(f".research/{self.research_name}/research.lock.json")
        assert not lock_file.exists(), "Lock file should be removed"


class TestStateManagement(TestWithTempDir):
    """Test cases for state management."""

    def setUp(self):
        super().setUp()

        self.research_name = "test-research"

    def test_save_and_load_state(self):
        """Test saving and loading state from disk."""
        research_dir = Path(f".research/{self.research_name}")
        research_dir.mkdir(parents=True, exist_ok=True)

        state = research.RWRState(
            original_topic="Test topic",
            breadth=3,
            depth=2,
            iteration=5,
            max_iterations=50,
            model="test/model",
            research_name=self.research_name,
        )

        research.save_state_to_disk(state)

        loaded = research.load_state_from_disk(self.research_name)

        assert loaded is not None, "Should load state successfully"
        assert loaded.original_topic == state.original_topic, "Topic should match"
        assert loaded.breadth == state.breadth, "Breadth should match"
        assert loaded.depth == state.depth, "Depth should match"
        assert loaded.iteration == state.iteration, "Iteration should match"
        assert loaded.max_iterations == state.max_iterations, "Max iterations should match"

    def test_load_state_nonexistent(self):
        """Test loading non-existent state returns None."""
        result = research.load_state_from_disk("nonexistent")
        assert result is None, "Should return None for non-existent state"

    def test_rwr_state_defaults(self):
        """Test RWRState default values."""
        state = research.RWRState()

        assert state.original_topic == ""
        assert state.breadth == research.DEFAULT_BREADTH
        assert state.depth == research.DEFAULT_DEPTH
        assert state.iteration == 0
        assert state.max_iterations == 0
        assert state.model == research.DEFAULT_MODEL
        assert state.review_model == research.DEFAULT_REVIEW_MODEL
        assert state.mock_mode == False


if __name__ == "__main__":
    unittest.main()
