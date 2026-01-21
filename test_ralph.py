#!/usr/bin/env python3
"""Test suite for ralph.py prompt generation system."""

import hashlib
import os
import secrets
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

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
        result = ralph.check_template_generation(self.state)

        assert result == True, (
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

        result = ralph.check_template_generation(self.state)

        assert result == False, (
            "Template generation should fail with malformed custom template",
            f"State: {self.state}, Result: {result}, Template: {malformed_template}"
        )


class TestArchiveSystem(unittest.TestCase):
    """Test cases for the archiving system."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.original_cwd = os.getcwd()
        cls.class_token = secrets.token_hex(16)
        cls.test_dir = Path(f"test_{cls.class_token}")
        cls.test_dir.mkdir(exist_ok=True)
        os.chdir(cls.test_dir)

        # Initialize git repository
        subprocess.run(["git", "init"], capture_output=True, check=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        os.chdir(cls.original_cwd)
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        ralph.ARCHIVED_HASHES.clear()

    def setUp(self):
        """Set up environment before each test."""
        self.lock_token = secrets.token_hex(16)
        ralph.ARCHIVED_HASHES.clear()

        # Ensure .ralph/archive/ directory exists
        archive_dir = Path(".ralph/archive")
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Check and reset git state if needed
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                capture_output=True,
                text=True,
                check=True
            )
            if "test commit" in result.stdout:
                subprocess.run(
                    ["git", "reset", "HEAD~1"],
                    capture_output=True,
                    check=True
                )
        except subprocess.CalledProcessError:
            pass

    def tearDown(self):
        """Clean up after each test."""
        ralph.ARCHIVED_HASHES.clear()

        # Check and reset git state if needed
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                capture_output=True,
                text=True,
                check=True
            )
            if "test commit" in result.stdout:
                subprocess.run(
                    ["git", "reset", "HEAD~1"],
                    capture_output=True,
                    check=True
                )
        except subprocess.CalledProcessError:
            pass

    def test_archive_intermediate_file_basic(self):
        """Test basic single file archiving."""
        test_file = Path("progress.md")
        test_file.write_text("Test content")

        ralph.archive_intermediate_file(test_file, self.lock_token)

        archive_dir = Path(".ralph/archive")
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

        assert ralph.ARCHIVE_FILE_FORMAT.match(archive_file.name), (
            "Archive filename should match expected format",
            f"Filename: {archive_file.name}"
        )

        assert archive_file.read_text() == "Test content", (
            "Archive content should match original file",
            f"Original: 'Test content', Archive: '{archive_file.read_text()}'"
        )

        expected_hash = hashlib.sha256(b"Test content").hexdigest()
        assert expected_hash in ralph.ARCHIVED_HASHES, (
            "File hash should be added to ARCHIVED_HASHES",
            f"Hash: {expected_hash}, ARCHIVED_HASHES: {ralph.ARCHIVED_HASHES}"
        )

    def test_archive_intermediate_file_deduplication(self):
        """Test hash-based deduplication prevents duplicate archives."""
        test_file = Path("progress.md")
        test_file.write_text("Test content")

        # Archive first time
        ralph.archive_intermediate_file(test_file, self.lock_token)
        archives_after_first = list(Path(".ralph/archive").glob(f"*.{self.lock_token}.progress.md"))

        # Archive second time (same content)
        ralph.archive_intermediate_file(test_file, self.lock_token)
        archives_after_second = list(Path(".ralph/archive").glob(f"*.{self.lock_token}.progress.md"))

        assert len(archives_after_first) == 1, (
            "First archive should create one file",
            f"Archives: {archives_after_first}"
        )

        assert len(archives_after_second) == 1, (
            "Second archive with same content should not create duplicate",
            f"Archives: {archives_after_second}"
        )

        # Sleep to ensure different timestamp for next archive
        time.sleep(1.5)

        # Modify content and archive again
        test_file.write_text("Modified content")

        # Clear hash to force new archive
        old_hash = hashlib.sha256(b"Test content").hexdigest()
        ralph.ARCHIVED_HASHES.discard(old_hash)

        ralph.archive_intermediate_file(test_file, self.lock_token)
        archives_after_third = list(Path(".ralph/archive").glob(f"*.{self.lock_token}.progress.md"))

        assert len(archives_after_third) == 2, (
            "Modified content should create new archive",
            f"Archives: {archives_after_third}"
        )

    def test_archive_intermediate_file_nonexistent(self):
        """Test graceful handling of missing files."""
        nonexistent_file = Path("does_not_exist.md")

        # Should not raise an error
        ralph.archive_intermediate_file(nonexistent_file, self.lock_token)

        archive_dir = Path(".ralph/archive")
        archives = list(archive_dir.glob(f"*.{self.lock_token}.does_not_exist.md"))

        assert len(archives) == 0, (
            "Should not create archive for nonexistent file",
            f"Archives: {archives}"
        )

        assert len(ralph.ARCHIVED_HASHES) == 0, (
            "ARCHIVED_HASHES should remain empty",
            f"ARCHIVED_HASHES: {ralph.ARCHIVED_HASHES}"
        )

    def test_archive_any_process_files(self):
        """Test batch archiving of all tracked process files."""
        # Create files from ARCHIVE_FILENAMES
        Path(ralph.PROGRESS_FILE).write_text("Progress content")
        Path(ralph.REQUEST_REVIEW_FILE).write_text("Review request")
        Path(ralph.REVIEW_PASSED_FILE).write_text("Review passed")

        # Create control file not in ARCHIVE_FILENAMES
        control_file = Path("ignored_file.txt")
        control_file.write_text("Ignored content")

        ralph.archive_any_process_files(self.lock_token)

        archive_dir = Path(".ralph/archive")

        # Check that ARCHIVE_FILENAMES files are archived
        archived_progress = list(archive_dir.glob(f"*.{self.lock_token}.progress.md"))
        archived_review_request = list(archive_dir.glob(f"*.{self.lock_token}.request.review.md"))
        archived_review_passed = list(archive_dir.glob(f"*.{self.lock_token}.review.passed.md"))

        assert len(archived_progress) == 1, (
            "progress.md should be archived",
            f"Archived progress files: {archived_progress}"
        )

        assert len(archived_review_request) == 1, (
            "request.review.md should be archived",
            f"Archived review request files: {archived_review_request}"
        )

        assert len(archived_review_passed) == 1, (
            "review.passed.md should be archived",
            f"Archived review passed files: {archived_review_passed}"
        )

        # Check that control file is NOT archived
        archived_control = list(archive_dir.glob(f"*.{self.lock_token}.ignored_file.txt"))
        assert len(archived_control) == 0, (
            "Ignored file should not be archived",
            f"Archived ignored files: {archived_control}"
        )

        # Original files should still exist (archive copies, doesn't delete)
        assert Path(ralph.PROGRESS_FILE).exists(), (
            "Original progress.md should still exist",
            "Original file was deleted"
        )

        assert control_file.exists(), (
            "Original ignored_file.txt should still exist",
            "Original control file was deleted"
        )

    def test_populate_archived_hashes(self):
        """Test hash loading from existing archives."""
        archive_dir = Path(".ralph/archive")

        # Create 3 archives with matching lock token
        for i in range(3):
            archive_name = f"1234567890.{self.lock_token}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        # Create 1 archive with different lock token
        different_token = secrets.token_hex(16)
        different_archive_name = f"1234567890.{different_token}.other.md"
        different_archive_path = archive_dir / different_archive_name
        different_archive_path.write_text("Other content")

        # Clear ARCHIVED_HASHES
        ralph.ARCHIVED_HASHES.clear()

        ralph.populate_archived_hashes(self.lock_token)

        assert len(ralph.ARCHIVED_HASHES) == 3, (
            "Should load exactly 3 hashes for matching token",
            f"ARCHIVED_HASHES count: {len(ralph.ARCHIVED_HASHES)}"
        )

        # Verify each hash matches SHA256 of corresponding content
        for i in range(3):
            expected_hash = hashlib.sha256(f"Content {i}".encode()).hexdigest()
            assert expected_hash in ralph.ARCHIVED_HASHES, (
                f"Hash for file{i} should be loaded",
                f"Expected hash: {expected_hash}"
            )

        # Different token's archive should NOT be loaded
        other_hash = hashlib.sha256(b"Other content").hexdigest()
        assert other_hash not in ralph.ARCHIVED_HASHES, (
            "Different token's hash should not be loaded",
            f"Other hash: {other_hash}"
        )

    def test_reorganize_archive_files_basic_with_git(self):
        """Test reorganization using git mv when git available."""
        archive_dir = Path(".ralph/archive")
        token_A = secrets.token_hex(16)

        # Create 3+ archives with token_A
        for i in range(3):
            archive_name = f"12345678{i}.{token_A}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        # Add files to git
        subprocess.run(["git", "add", "."], capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "test commit"],
            capture_output=True,
            check=True
        )

        # Call reorganize with different active token
        token_B = secrets.token_hex(16)
        ralph.reorganize_archive_files(token_B)

        # Assert files moved to session directory
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

        # Verify files are tracked in git
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True
        )
        assert token_A in result.stdout, (
            "Moved files should be tracked in git",
            f"Git status: {result.stdout}"
        )

        # Verify archive parent directory is clean for token_A files
        token_A_files_in_parent = list(archive_dir.glob(f"*.{token_A}.*"))
        assert len(token_A_files_in_parent) == 0, (
            "Token_A files should not remain in parent directory",
            f"Files in parent: {token_A_files_in_parent}"
        )

    def test_reorganize_archive_files_fallback_to_copy(self):
        """Test fallback to shutil.copy2 when git mv fails."""
        archive_dir = Path(".ralph/archive")
        token_A = secrets.token_hex(16)

        # Create 3+ archives with token_A
        for i in range(3):
            archive_name = f"12345678{i}.{token_A}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        # Do NOT add/commit to git (files untracked)
        # This will cause git mv to fail, triggering copy2 fallback

        # Call reorganize with different active token
        token_B = secrets.token_hex(16)
        ralph.reorganize_archive_files(token_B)

        # Assert files moved using copy2 fallback
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

        # Verify original files deleted
        token_A_files_in_parent = list(archive_dir.glob(f"*.{token_A}.*"))
        assert len(token_A_files_in_parent) == 0, (
            "Original files should be deleted after copy2",
            f"Files in parent: {token_A_files_in_parent}"
        )

    def test_reorganize_archive_files_skips_single_files(self):
        """Test that single-file sessions are NOT reorganized."""
        archive_dir = Path(".ralph/archive")
        token_A = secrets.token_hex(16)
        token_B = secrets.token_hex(16)

        # Create exactly 1 archive for token_A
        single_archive_name = f"1234567890.{token_A}.single.md"
        single_archive_path = archive_dir / single_archive_name
        single_archive_path.write_text("Single file content")

        # Create 3+ archives for token_B
        for i in range(3):
            archive_name = f"12345678{i}.{token_B}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        # Call reorganize with different active token
        token_C = secrets.token_hex(16)
        ralph.reorganize_archive_files(token_C)

        # Assert token_A single file NOT moved
        token_A_session_dir = archive_dir / token_A
        assert not token_A_session_dir.exists(), (
            "Single-file session should NOT be reorganized",
            f"Session dir created: {token_A_session_dir}"
        )

        # Assert token_A file still in parent directory
        token_A_files_in_parent = list(archive_dir.glob(f"*.{token_A}.*"))
        assert len(token_A_files_in_parent) == 1, (
            "Single file should remain in parent directory",
            f"Token_A files in parent: {token_A_files_in_parent}"
        )

        # Assert token_B files ARE moved
        token_B_session_dir = archive_dir / token_B
        assert token_B_session_dir.exists(), (
            "Multi-file session should be reorganized",
            f"Session dir not found: {token_B_session_dir}"
        )

        token_B_files_in_parent = list(archive_dir.glob(f"*.{token_B}.*"))
        assert len(token_B_files_in_parent) == 0, (
            "Token_B files should be moved from parent directory",
            f"Token_B files in parent: {token_B_files_in_parent}"
        )

    def test_reorganize_archive_files_skips_active_session(self):
        """Test that active session files are NOT reorganized."""
        archive_dir = Path(".ralph/archive")
        active_token = secrets.token_hex(16)

        # Create archives with active_token
        for i in range(3):
            archive_name = f"12345678{i}.{active_token}.file{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Content {i}")

        # Call reorganize with same active token
        ralph.reorganize_archive_files(active_token)

        # Assert active session files NOT moved
        active_session_dir = archive_dir / active_token
        assert not active_session_dir.exists(), (
            "Active session files should NOT be reorganized",
            f"Session dir created: {active_session_dir}"
        )

        # Assert files remain in parent directory
        active_token_files = list(archive_dir.glob(f"*.{active_token}.*"))
        assert len(active_token_files) == 3, (
            "Active session files should remain in parent directory",
            f"Active token files: {active_token_files}"
        )

    def test_reorganize_archive_files_duplicate_session_dirs(self):
        """Test handling of existing session directories."""
        archive_dir = Path(".ralph/archive")
        token_A = secrets.token_hex(16)

        # Create existing session directory
        existing_session_dir = archive_dir / token_A
        existing_session_dir.mkdir(exist_ok=True)
        (existing_session_dir / "existing.md").write_text("Existing file")

        # Create new archives for same token_A
        for i in range(2):
            archive_name = f"12345678{i}.{token_A}.newfile{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"New content {i}")

        # Call reorganize with different active token
        token_B = secrets.token_hex(16)
        ralph.reorganize_archive_files(token_B)

        # Assert creates token_A_1/ to avoid conflict
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

        # Test with existing _1, should create _2
        existing_session_dir_1 = archive_dir / f"{token_A}_1"
        existing_session_dir_1.mkdir(exist_ok=True)
        (existing_session_dir_1 / "existing.md").write_text("Existing file 1")

        # Create more archives for same token_A
        for i in range(2, 4):
            archive_name = f"12345678{i}.{token_A}.newerfile{i}.md"
            archive_path = archive_dir / archive_name
            archive_path.write_text(f"Newer content {i}")

        token_C = secrets.token_hex(16)
        ralph.reorganize_archive_files(token_C)

        # Assert creates token_A_2/
        newer_session_dir = archive_dir / f"{token_A}_2"
        assert newer_session_dir.exists(), (
            "Should create {token_A}_2/ to avoid conflict with _1",
            f"Archive dir contents: {list(archive_dir.iterdir())}"
        )

    def test_reorganize_archive_files_no_archives(self):
        """Test graceful handling when no archives exist."""
        archive_dir = Path(".ralph/archive")

        # Ensure archive directory is empty or doesn't exist
        if archive_dir.exists():
            shutil.rmtree(archive_dir)

        # Call reorganize - should not raise error
        token_A = secrets.token_hex(16)
        ralph.reorganize_archive_files(token_A)

        # Assert no directories created
        subdirs = [d for d in archive_dir.iterdir() if d.is_dir()] if archive_dir.exists() else []
        assert len(subdirs) == 0, (
            "Should not create any directories when no archives exist",
            f"Subdirectories: {subdirs}"
        )

        # Assert archive directory still clean
        assert not archive_dir.exists() or len(list(archive_dir.iterdir())) == 0, (
            "Archive directory should be clean",
            f"Archive dir contents: {list(archive_dir.iterdir()) if archive_dir.exists() else []}"
        )

    def test_archive_timestamp_ordering(self):
        """Test that timestamps are monotonically increasing."""
        # Clean archive directory from previous tests
        archive_dir = Path(".ralph/archive")
        if archive_dir.exists():
            for file in archive_dir.iterdir():
                file.unlink()

        test_file1 = Path("file1.md")
        test_file1.write_text("Content 1")

        test_file2 = Path("file2.md")
        test_file2.write_text("Content 2")

        # Archive first file
        ralph.archive_intermediate_file(test_file1, self.lock_token)
        archives_before_sleep = list(Path(".ralph/archive").glob(f"*.{self.lock_token}.file1.md"))
        timestamp1 = int(archives_before_sleep[0].name.split('.')[0])

        # Use 1.5 second delay
        time.sleep(1.5)

        # Archive second file
        ralph.archive_intermediate_file(test_file2, self.lock_token)
        archives_after_sleep = list(Path(".ralph/archive").glob(f"*.{self.lock_token}.file2.md"))
        timestamp2 = int(archives_after_sleep[0].name.split('.')[0])

        assert timestamp1 < timestamp2, (
            "Timestamps should be monotonically increasing",
            f"timestamp1: {timestamp1}, timestamp2: {timestamp2}, difference: {timestamp2 - timestamp1}"
        )

        # Verify both archives have same lock_token
        assert archives_before_sleep[0].name.split('.')[1] == archives_after_sleep[-1].name.split('.')[1], (
            "Both archives should have the same lock_token",
            f"Token1: {archives_before_sleep[0].name.split('.')[1]}, Token2: {archives_after_sleep[-1].name.split('.')[1]}"
        )

    def test_archive_format_regex_validation(self):
        """Test that all generated archives match expected format."""
        test_files = ["file1.md", "file2.txt", "file3.log"]
        test_files = [Path(f) for f in test_files]
        for i, test_file in enumerate(test_files):
            test_file.write_text(f"Content {i}")
            ralph.archive_intermediate_file(test_file, self.lock_token)

        archives = list(Path(".ralph/archive").glob(f"*.{self.lock_token}.*"))

        for archive in archives:
            match = ralph.ARCHIVE_FILE_FORMAT.match(archive.name)
            assert match is not None, (
                f"Archive filename should match expected format",
                f"Filename: {archive.name}, Pattern: {ralph.ARCHIVE_FILE_FORMAT.pattern}"
            )

            timestamp_str, lock_token, original_filename = match.groups()

            # Validate lock_token is exactly 32 hex chars
            assert len(lock_token) == 32, (
                f"Lock token should be exactly 32 hex characters",
                f"Lock token length: {len(lock_token)}, value: {lock_token}"
            )

            assert all(c in "0123456789abcdef" for c in lock_token), (
                f"Lock token should be valid hex string",
                f"Lock token: {lock_token}"
            )

            # Validate timestamp is valid integer
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

            # Validate original_filename matches one of our test files
            assert original_filename in ["file1.md", "file2.txt", "file3.log"], (
                f"Original filename should match test files",
                f"Original filename: {original_filename}"
            )


if __name__ == "__main__":
    unittest.main()
