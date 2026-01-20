#!/usr/bin/env python3
"""
Enhanced RWL Implementation

A self-contained Python CLI tool that orchestrates AI agent development
through a structured four-phase cycle: BUILD-REVIEW-PLAN-COMMIT.

Usage:
    python ralph.py "Your task description"
    python ralph.py --enhanced --final-review "Complex task"
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# semver string
VERSION = "0.0.4"


# Configuration Constants
DEFAULT_MODEL = "opencode/big-pickle"
DEFAULT_REVIEW_MODEL = "opencode/big-pickle"
OPENCODE_TIMEOUT = 1200 # 20 minutes
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_REVIEW_EVERY = 0
PHASE_RETRY_LIMIT = 3
RETRY_WAIT_SECONDS = 30
RECOVERY_WAIT_SECONDS = 10
STALE_LOCK_TIMEOUT = 3600 # 60 minutes


class Phase(Enum):
    BUILD = "BUILD"
    REVIEW = "REVIEW"
    PLAN = "PLAN"
    COMMIT = "COMMIT"
    RECOVERY = "RECOVERY"


@dataclass
class RWLState:
    """Represents the current state of the RWL process."""
    iteration: int = field(default=0)
    current_phase: str = field(default=Phase.BUILD.value)
    failed_phase: str | None = field(default=None)
    retry_count: int = field(default=0)
    max_iterations: int = field(default=DEFAULT_MAX_ITERATIONS)
    enhanced_mode: bool = field(default=False)
    skip_tests: bool = field(default=False)
    review_every: int = field(default=0)
    model: str = field(default=DEFAULT_MODEL)
    review_model: str = field(default=DEFAULT_REVIEW_MODEL)
    lock_token: str | None = field(default=None)
    start_time: float = field(default=0.0)
    phase_history: list[str] = field(default_factory=list)
    last_error: str | None = field(default=None)
    final_review_requested: bool = field(default=False)
    is_complete: bool = field(default=False)
    mock_mode: bool = field(default=False)
    push_commits: bool = field(default=False)
    original_prompt: str = field(default="")
    phase_recovered: bool = field(default=False)


def setup_signal_handlers(state: RWLState) -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum: int, frame) -> None:
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        release_project_lock(state.lock_token)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_logging() -> None:
    """Set up logging directory structure."""
    logs_dir = Path(".ralph/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)


# Global state for archiving
ARCHIVED_HASHES: set[str] = set()

ARCHIVE_FILENAMES = [
    "request.review.md",
    "review.passed.md",
    "review.rejected.md",
    "review.final.md",
    "plan.review.md",
    "implementation_plan.md",
    "progress.md",
    "recovery.notes.md",
    "completed.md",
]

ARCHIVE_FILE_FORMAT = re.compile(r"^(\d+)\.([a-f0-9]{32})\.(.+)$")


def populate_archived_hashes(lock_token: str) -> None:
    """Populate the in-memory hash set from existing archives for this token."""
    archive_dir = Path(".ralph/archive")
    if not archive_dir.exists():
        return

    for archive_file in archive_dir.glob(f"*.{lock_token}.*"):
        try:
            content = archive_file.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            ARCHIVED_HASHES.add(file_hash)
        except Exception as e:
            print(f"WARNING: Could not read archive file {archive_file}: {e}")


def archive_intermediate_file(file_path: Path, lock_token: str) -> None:
    """Archive an intermediate file with a timestamp prefix if it hasn't been archived yet."""
    if not file_path.exists():
        return

    try:
        content = file_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        if file_hash in ARCHIVED_HASHES:
            return

        archive_dir = Path(".ralph/archive")
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        archive_name = f"{timestamp}.{lock_token}.{file_path.name}"
        archive_path = archive_dir / archive_name

        shutil.copy2(file_path, archive_path)
        ARCHIVED_HASHES.add(file_hash)
        print(f"Archived: {file_path.name} -> {archive_name}")
    except Exception as e:
        print(f"WARNING: Could not archive {file_path}: {e}")


def archive_any_process_files(lock_token: str) -> None:
    """Check for and archive any intermediate files."""
    for file_name in ARCHIVE_FILENAMES:
        archive_intermediate_file(Path(file_name), lock_token)


def reorganize_archive_files(lock_token: str | None) -> None:
    """Reorganize archives into per-session directories.

    Skips files matching the current lock token to allow active sessions to continue.
    Only reorganizes sessions with 2 or more files.
    """
    archive_dir = Path(".ralph/archive")
    if not archive_dir.exists():
        return

    current_token = lock_token

    token_files: dict[str, list[Path]] = {}

    for archive_file in archive_dir.iterdir():
        if not archive_file.is_file():
            continue

        match = ARCHIVE_FILE_FORMAT.match(archive_file.name)
        if match:
            timestamp_str, token, original_name = match.groups()
            if current_token and token == current_token:
                continue
            if token not in token_files:
                token_files[token] = []
            token_files[token].append(archive_file)

    for token, files in token_files.items():
        if len(files) < 2:
            continue

        session_dir = archive_dir / token
        if session_dir.exists():
            suffix = 1
            while (archive_dir / f"{token}_{suffix}").exists():
                suffix += 1
            session_dir = archive_dir / f"{token}_{suffix}"

        session_dir.mkdir(exist_ok=True)

        for src_file in files:
            dst_file = session_dir / src_file.name
            try:
                result = subprocess.run(
                    ["git", "mv", str(src_file), str(dst_file)],
                    capture_output=True,
                    cwd=Path.cwd()
                )
                if result.returncode != 0:
                    raise Exception(result.stderr.decode().strip())
            except Exception:
                shutil.copy2(src_file, dst_file)
                src_file.unlink()


def validate_environment() -> None:
    """Validate the environment before starting."""
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("ERROR: This directory is not a git repository")
        sys.exit(1)

    # Check if we can create the .ralph directory
    ralph_dir = Path(".ralph")
    try:
        ralph_dir.mkdir(exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create .ralph directory: {e}")
        sys.exit(1)


def get_project_lock() -> str | None:
    """Create a project lock file and return the lock token."""
    lock_file = Path("ralph.lock.json")

    # Check for existing lock
    if lock_file.exists():
        try:
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            # Check if lock is stale (older than 1 hour)
            created_time = lock_data.get("created", 0)
            if time.time() - created_time < STALE_LOCK_TIMEOUT:  # 1 hour
                print("ERROR: Project is locked by another RWL instance")
                return None
            else:
                print("WARNING: Removing stale lock file")
                lock_file.unlink()
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"WARNING: Invalid lock file format, removing: {e}")
            try:
                lock_file.unlink()
            except:
                pass

    # Create new lock
    import secrets
    token = secrets.token_hex(16)
    timestamp = time.time()
    pid = os.getpid()

    lock_data = {
        "lock_token": token,
        "created": timestamp,
        "pid": pid
    }

    try:
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)
        return token
    except Exception as e:
        print(f"ERROR: Cannot create lock file: {e}")
        return None


def check_project_lock(token: str) -> bool:
    """Check if the project lock is still valid."""
    lock_file = Path("ralph.lock.json")

    if not lock_file.exists():
        return False

    try:
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        valid = lock_data.get("lock_token") == token
        # update the timestamp if it is our lock, then return
        if valid:
            lock_data["created"] = time.time()
            with open(lock_file, "w") as f:
                json.dump(lock_data, f)
        return valid
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def release_project_lock(token: str | None) -> None:
    """Release the project lock."""
    if not token:
        return

    lock_file = Path("ralph.lock.json")

    try:
        if lock_file.exists():
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            if lock_data.get("lock_token") == token:
                lock_file.unlink()
                print("Project lock released")
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"WARNING: Error releasing lock: {e}")


def cleanup_process_files(lock_token: str) -> None:
    """Clean up lingering process files after successful completion."""
    archive_any_process_files(lock_token)
    files_to_remove = [
        "request.review.md",
        "review.rejected.md",
        "review.passed.md",
        "review.final.md",
        "progress.md",
        "recovery.notes.md"
    ]

    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            try:
                path.unlink()
                print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"WARNING: Could not remove {file_path}: {e}")


def save_state_to_disk(state: RWLState) -> None:
    """Save the current state to disk for crash recovery."""
    state_file = Path(".ralph/state.json")

    try:
        # Convert dataclass to dictionary, handling None values properly
        state_dict = dataclasses.asdict(state)

        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
    except Exception as e:
        print(f"WARNING: Error saving state: {e}")


def load_state_from_disk() -> RWLState | None:
    """Load state from disk for crash recovery."""
    state_file = Path(".ralph/state.json")

    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            state_dict = json.load(f)

        if not state_dict.get("original_prompt"):
            print("ERROR: Saved state has no original prompt. Cannot resume.")
            return None

        return RWLState(**state_dict)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"ERROR: Corrupted state file: {e}")
        return None


def generate_initial_plan_prompt(state: RWLState, cli_prompt: str = "") -> str:
    """Generate the initial implementation plan prompt with optional CLI prompt prepended."""
    combined = f"{cli_prompt}\n{state.original_prompt}".strip()

    return f"""You are creating an initial implementation plan.

    TASK:
    {combined}

    INSTRUCTIONS:
    1. Analyze the task and break it down into concrete implementation steps
    2. Create implementation_plan.md with prioritized tasks
    3. Focus on a logical progression from foundation to completion

    TASK PLANNING:
    - Break the task into manageable sub-tasks
    - Identify dependencies between tasks
    - Prioritize tasks that enable subsequent work
    - Consider testing, documentation, and software engineering standards
    - Do NOT over-specify with irrelevant information

    PLAN FORMAT:
    # Implementation Plan

    ## Tasks

    ### TASK_NAME

    - Status: Pending
    - Description: TASK_DESCRIPTION
    - Acceptance Criteria:
        - CRITERION_1
        - CRITERION_2
        - CRITERION_3

    ## Dependencies
    Note any task dependencies or prerequisites

    OUTPUT: Create implementation_plan.md with your plan"""


def generate_plan_review_prompt(state: RWLState) -> str:
    return f"""You are reviewing an implementation plan.

    ORIGINAL PROMPT:
    {state.original_prompt}

    INSTRUCTIONS:
    1. Read the plan written to implementation_plan.md.
    2. Analyze the plan for consistency with the requirements of the original prompt.
    3. Analyze the plan for coherence.
    4. Analyze the plan for format: each task must have a Status, Description, and
    Acceptance Criteria.
    6. Note that the plan need not be exhaustively detailed. It needs only to have the
    level of detail required to guide an implementer toward the ultimate goals of the
    original prompt.
    7. Focus on the 2-3 most glaring issues.
    8. Provide constructive criticism.

    OUTPUT: Create plan.review.md with your analysis. If the plan does not need any
    updates, skip this step."""


def generate_revise_plan_prompt(state: RWLState) -> str:
    return f"""You are revising an implementation plan to incorporate and address
    the critique from the reviewer.

    ORIGINAL PROMPT:
    {state.original_prompt}

    INSTRUCTIONS:
    1. Read the plan written to implementation_plan.md.
    2. Read the analysis/critique provided in plan.review.md.
    3. Consider carefully what parts of the analysis/critique are valid and which
    are not.
    4. If there were any valid points to the analysis/critique, update
    implementation_plan.md
    to address those concerns/incorporate that feedback.
    5. Do NOT over-specify with irrelevant details. Leave something to the
    judgment of the implementer.
    6. Delete the review.plan.md file when you are done."""


def generate_build_prompt(state: RWLState, active_task: str|None = None) -> str:
    """Generate the prompt for the BUILD phase."""
    return f"""You are a software engineer working to complete programming tasks.

    ORIGINAL PROMPT:
    {state.original_prompt}

    PHASE: BUILD
    Your goal is to choose a task from the implementation_plan.md file and work
    toward completing it.

    CONTEXT:
    - Current iteration: {state.iteration}
    - Max iterations: {'not limited' if state.max_iterations == 0
    else state.max_iterations}
    - {'You must NOT run tests' if state.skip_tests
    else 'You should run tests to validate your work'}

    INSTRUCTIONS:
    1. Read the implementation_plan.md file to identify {'a task that has not\
    been completed' if not active_task else ('the active task: ' + active_task)}
    2. Review progress.md, looking for information relevant for your chosen task
    3. Implement the task following best practices
    4. Write what you learned, what you struggled with, and what remains to be
    done for this task in progress.md
    5. Stage your changes with 'git add'
    6. {'Create request.review.md when the task is complete'
    if state.enhanced_mode else 'Halt'}

    TASK SELECTION:
    - Choose the highest priority task from implementation_plan.md
    - Mark tasks you are working on as "In Progress"
    - Mark completed tasks as "In Review" in the implementation_plan.md file
    - Focus on one task at a time

    PROGRESS TRACKING:
    - Update progress.md with: learnings, struggles, remaining work
    - If all tasks are marked as "Done" or "Complete" in implementation_plan.md,
    create completed.md with exactly: <promise>COMPLETE</promise>
    IMPORTANT: """ + (
        'The previous attempt to run this phase failed. Read \
        recovery.notes.md for recovery information.'
        if state.phase_recovered else
        'Focus on quality implementation and clear documentation.'
    )


def generate_final_build_prompt(state: RWLState, active_task: str|None = None) -> str:
    """Generate the prompt for the BUILD phase."""
    return f"""You are putting the finishing touches on the project.

    ORIGINAL PROMPT:
    {state.original_prompt}

    PHASE - FINAL BUILD:
    Your task is to read the review.final.md file for feedback on issues that need
    to be addressed, then work to address them."""


def generate_review_prompt(state: RWLState, is_final_review: bool = False) -> str:
    """Generate the prompt for the REVIEW phase."""
    if is_final_review:
        return f"""You are conducting a FINAL REVIEW of the completed implementation.

        ORIGINAL PROMPT:
        {state.original_prompt}

        PHASE - FINAL REVIEW:
        Your goal is to ensure the implementation is polished and ready for delivery.

        INSTRUCTIONS:
        1. Review all implemented code for quality, coherence, consistency with the task
        description/goal/acceptance criteria, and completeness.
        2. Do not nitpick small, irrelevant issues. Focus on what really matters:
        completion
        of the task requirements.
        3. If you do find an issue with polish, professionalism, or code style, only point
        it out if the fix will not require a major refactor or cause more harm than good.
        3. Identify any remaining issues or improvements that are needed
        4. Create review.final.md with your findings

        OUTPUT:
        - If everything is satisfactory: create review.final.md with the a concise
        analysis of work completed
        - If improvements are needed: create review.final.md with specific action items"""
    else:
        prompt = f"""You are reviewing the completion of a task in the development cycle.

        ORIGINAL PROMPT:
        {state.original_prompt}

        PHASE: REVIEW
        Your goal is to evaluate if the completed task meets quality standards.

        INSTRUCTIONS:
        1. Read request.review.md to understand what was completed
        2. Review the implementation changes (check git status, diff staged files)
        3. Evaluate against project requirements and best practices
        4. Create either review.passed.md OR review.rejected.md

        REVIEW CRITERIA:
        - Task completion: Is the task fully implemented? This is the most important
        criterion. All else pales in comparison to consistency with the task description,
        requirements, and acceptance criteria.
        - Code quality: Does it follow best practices?
        - Testing: Are appropriate and relevant tests included?
        - Documentation: Does the code have sufficient type annotations and
        docblocks/docstrings?

        OUTPUT FORMAT:
        - If PASSED:
            - Create review.passed.md with approval and any minor suggestions
            - Update the status of any relevant tasks in the implementation_plan.md
            file from "In Review" to "Done"
        - If REJECTED:
            - Create review.rejected.md with specific issues and action items
            - Update the status of the rejected tasks in the implementation_plan.md
            file from "In Review" to "In Progress"

        CRITICAL: You must create exactly one of review.passed.md or review.rejected.md."""

    if state.phase_recovered:
        return prompt + """

        The previous attempt to run this phase failed. Read recovery.notes.md for recovery information."""
    else:
        return prompt


def generate_plan_prompt(state: RWLState) -> str:
    """Generate the prompt for the PLAN phase."""
    prompt = f"""You are planning development tasks or revising an existing plan.

    ORIGINAL PROMPT:
    {state.original_prompt}

    PHASE: PLAN
    Your goal is to update the implementation plan based on current progress and feedback.

    CONTEXT:
    - Current iteration: {state.iteration}
    - Max iterations: {'not limited' if state.max_iterations == 0
    else state.max_iterations}

    INSTRUCTIONS:
    1. Read review.rejected.md if it exists (for feedback on rejected work)
    2. Read progress.md if it exists (for learnings and struggles)
    3. Read the current implementation_plan.md
    4. Update implementation_plan.md with prioritized tasks

    TASK PRIORITIZATION:
    - Address any issues from review.rejected.md first
    - Focus on remaining implementation work
    - Consider dependencies between tasks
    - Note any blockers

    PLAN FORMAT:
    # Implementation Plan

    ## Tasks

    ### TASK_NAME

    - Status: Pending
    - Description: TASK_DESCRIPTION
    - Acceptance Criteria:
        - CRITERION_1
        - CRITERION_2
        - CRITERION_3

    ## Dependencies
    Note any task dependencies or prerequisites

    OUTPUT: Update implementation_plan.md with the revised plan"""

    if state.phase_recovered:
        return prompt + """

        The previous attempt to run this phase failed. Read recovery.notes.md for recovery information."""
    else:
        return prompt


def generate_commit_prompt(state: RWLState) -> str:
    """Generate the prompt for the COMMIT phase."""
    prompt = f"""You are preparing to commit completed work.

    PHASE: COMMIT
    Your goal is to stage appropriate files and create a meaningful commit message.

    INSTRUCTIONS:
    1. Read review.passed.md for review feedback
    2. Check git status to see what files are staged/unstaged
    3. Stage additional files as needed for a complete commit
    4. Create a conventional commit message incorporating review feedback

    STAGING STRATEGY:
    - Include all files related to the completed task
    - Exclude temporary files, logs, or intermediate artifacts
    - Ensure the commit tells a complete story

    COMMIT MESSAGE FORMAT:
    - Use conventional commit format: type(scope): description
    - Examples: feat(auth): add user authentication
    - Examples: fix(api): resolve null pointer exception
    - Examples: docs(readme): update installation instructions
    - Include review feedback insights in the message

    ACTIONS:
    1. Stage additional files if needed: git add <files>
    2. Execute the commit with the generated message
    3. Commit locally only (pushing can be configured separately)"""

    if state.phase_recovered:
        return prompt + """

        The previous attempt to run this phase failed. Read recovery.notes.md for recovery information."""
    else:
        return prompt


def generate_recovery_prompt(state: RWLState, failed_phase: str, error: str) -> str:
    """Generate the prompt for the RECOVERY phase."""
    return f"""You are diagnosing and recovering from a phase failure.

    ORIGINAL PROMPT:
    {state.original_prompt}

    PHASE: RECOVERY
    Failed Phase: {failed_phase}
    Error: {error}

    CONTEXT:
    - Current iteration: {state.iteration}
    - Max iterations: {'not limited' if state.max_iterations == 0
    else state.max_iterations}
    - Retry count: {state.retry_count}

    INSTRUCTIONS:
    1. Analyze what went wrong in the failed phase
    2. Review .ralph/state.json and .ralph/logs/ for diagnostic information
    3. Identify the root cause of the failure
    4. Create a recovery plan in recovery.notes.md

    DIAGNOSTIC STEPS:
    - Check for file system issues (permissions, disk space)
    - Verify git repository state
    - Look for configuration or environment problems
    - Check for timeout or network issues

    RECOVERY OUTPUT (recovery.notes.md):
    - Root cause analysis
    - If it was a timeout:
        - Provide guidance for continuing where the previous phase left off
    - If it was NOT a timeout:
        - Specific steps to resolve the issue
        - Prevention measures for future iterations
        - Whether to retry the failed phase or continue with adjustments

    GOAL: Provide actionable recovery guidance to get development back on track."""


def call_opencode(prompt: str, model: str, lock_token: str, timeout: int = OPENCODE_TIMEOUT, mock_mode: bool = False) -> tuple[bool, str]:
    """Call OpenCode with the given prompt and return success status and output."""
    if mock_mode:
        # Mock mode for testing
        return True, f"MOCK RESPONSE for: {prompt[:100]}..."

    try:
        cmd = ["opencode", "run", "--model", model]

        print(f"Executing: opencode --model {model}")
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd()
        )

        archive_any_process_files(lock_token)

        if result.returncode == 0:
            return True, result.stdout
        else:
            error_msg = f"OpenCode failed with code {result.returncode}: {result.stderr}"
            print(f"ERROR: {error_msg}")
            return False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = f"OpenCode timed out after {timeout} seconds"
        print(f"ERROR: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error calling OpenCode: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def generate_initial_plan(state: RWLState, cli_prompt: str = "") -> tuple[bool, str]:
    """Generate the initial implementation plan when none exists."""
    initial_plan_prompt = generate_initial_plan_prompt(state, cli_prompt)
    plan_review_prompt = generate_plan_review_prompt(state)
    revise_plan_prompt = generate_revise_plan_prompt(state)

    # Execute the plan generation
    success, result = call_opencode(
        initial_plan_prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode
    )
    total_result = result
    if not success:
        return False, result

    # Now enter plan revision loop
    for _ in range(2):
        success, result = call_opencode(
            plan_review_prompt, state.review_model, state.lock_token or "", mock_mode=state.mock_mode
        )
        total_result += f'\n\n{result}'
        if not success:
            return False, result
        if not Path('plan.review.md').exists():
            return True, result
        success, result = call_opencode(
            revise_plan_prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode
        )
        total_result += f'\n\n{result}'
        if not success:
            return False, total_result

    return True, total_result


def execute_build_phase(state: RWLState) -> tuple[bool, str]:
    """Execute the BUILD phase."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print(f"Starting BUILD phase")

    try:
        # Generate build prompt
        prompt = generate_build_prompt(state)

        # Execute via OpenCode
        success, result = call_opencode(prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode)

        if success:
            print("BUILD phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"BUILD phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_final_build_phase(state: RWLState) -> tuple[bool, str]:
    """Execute the final BUILD phase."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print(f"Starting final BUILD phase")

    try:
        # Generate build prompt
        prompt = generate_final_build_prompt(state)

        # Execute via OpenCode
        success, result = call_opencode(prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode)

        if success:
            print("Final BUILD phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"Final BUILD phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_review_phase(state: RWLState, is_final_review: bool = False) -> tuple[bool, str]:
    """Execute the REVIEW phase."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print("Starting REVIEW phase")

    try:
        # Wait for request.review.md in enhanced mode (for regular reviews)
        if not is_final_review and state.enhanced_mode:
            request_file = Path("request.review.md")
            if not request_file.exists():
                print(f"Waiting {RETRY_WAIT_SECONDS} seconds for request.review.md...")
                time.sleep(RETRY_WAIT_SECONDS)
                if not request_file.exists():
                    return False, "request.review.md file not found"

        # Generate review prompt
        prompt = generate_review_prompt(state, is_final_review)

        # Execute via OpenCode
        success, result = call_opencode(prompt, state.review_model, state.lock_token or "", mock_mode=state.mock_mode)

        if success:
            # Verify that exactly one of the required files was created
            if is_final_review:
                expected_file = "review.final.md"
                created = Path(expected_file).exists()
                if created:
                    print("Final REVIEW phase completed successfully")
                    return True, result
                else:
                    return False, f"Expected {expected_file} was not created"
            else:
                passed_exists = Path("review.passed.md").exists()
                rejected_exists = Path("review.rejected.md").exists()

                if passed_exists ^ rejected_exists:  # XOR - exactly one exists
                    print("REVIEW phase completed successfully")
                    return True, result
                else:
                    return False, "REVIEW phase must create exactly one of: review.passed.md, review.rejected.md"
        else:
            return False, result

    except Exception as e:
        error_msg = f"REVIEW phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_plan_phase(state: RWLState) -> tuple[bool, str]:
    """Execute the PLAN phase."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print("Starting PLAN phase")

    try:
        # Generate plan prompt
        prompt = generate_plan_prompt(state)

        # Execute via OpenCode
        success, result = call_opencode(prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode)

        if success:
            # Clean up review.rejected.md if it existed
            if Path("review.rejected.md").exists():
                Path("review.rejected.md").unlink()

            print("PLAN phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"PLAN phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_commit_phase(state: RWLState) -> tuple[bool, str]:
    """Execute the COMMIT phase."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print("Starting COMMIT phase")

    try:
        if not Path("review.passed.md").exists():
            return False, "review.passed.md not found"

        # Generate commit prompt
        prompt = generate_commit_prompt(state)

        # Execute via OpenCode
        success, result = call_opencode(prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode)

        if success:
            # Clean up review.passed.md
            if Path("review.passed.md").exists():
                Path("review.passed.md").unlink()

            print("COMMIT phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"COMMIT phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_recovery_phase(state: RWLState, failed_phase: str, error: str) -> tuple[bool, str]:
    """Execute the RECOVERY phase."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print(f"Starting RECOVERY phase for {failed_phase}")

    try:
        # Wait briefly for transient issues to resolve
        time.sleep(RECOVERY_WAIT_SECONDS)

        # Generate recovery prompt
        prompt = generate_recovery_prompt(state, failed_phase, error)

        # Execute via OpenCode
        success, result = call_opencode(prompt, state.model, state.lock_token or "", mock_mode=state.mock_mode)

        if success:
            print("RECOVERY phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"RECOVERY phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_final_review_phase(state: RWLState) -> tuple[bool, str]:
    """Execute the final REVIEW phase."""
    return execute_review_phase(state, is_final_review=True)


def should_trigger_review(state: RWLState) -> bool:
    """Determine if REVIEW phase should be triggered."""
    # Enhanced mode: check for request.review.md file
    if state.enhanced_mode and Path("request.review.md").exists():
        return True

    # Check review-every parameter
    if state.review_every > 0 and state.iteration > 0:
        return state.iteration % state.review_every == 0

    return False


def retry_failed_phase(state: RWLState, failed_phase: str) -> tuple[bool, str]:
    """Retry the specific failed phase using the same execute functions."""
    if failed_phase == Phase.BUILD.value:
        return execute_build_phase(state)
    elif failed_phase == Phase.REVIEW.value:
        return execute_review_phase(state)
    elif failed_phase == Phase.PLAN.value:
        return execute_plan_phase(state)
    elif failed_phase == Phase.COMMIT.value:
        return execute_commit_phase(state)
    else:
        return False, f"Unknown phase for retry: {failed_phase}"


def handle_phase_failure(state: RWLState, failed_phase: str, error: str) -> tuple[bool, int]:
    """Handle phase failure with retry logic and recovery."""
    print(f"ERROR: {failed_phase} phase failed: {error}")
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    state.failed_phase = failed_phase
    state.last_error = error

    # Log the failure
    log_path = Path(f".ralph/logs/errors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{time.time()}: {failed_phase} failed: {error}\n")

    # Reset retry count and start recovery loop
    state.retry_count = 0
    recovery_success, recovery_result = False, None

    # Enhanced mode: try recovery and retry until limit reached
    while state.enhanced_mode and state.retry_count < PHASE_RETRY_LIMIT:
        # Increment retry count at start of loop
        state.retry_count += 1

        if not recovery_success:
            print(f"Attempting RECOVERY phase for {failed_phase} (attempt {state.retry_count}/{PHASE_RETRY_LIMIT})...")
            recovery_success, recovery_result = execute_recovery_phase(state, failed_phase, error)

        if recovery_success:
            # Set phase_recovered flag for prompt generation
            state.phase_recovered = True
            print(f"Recovery successful, retrying {failed_phase} (attempt {state.retry_count}/{PHASE_RETRY_LIMIT})...")

            # Retry the failed phase
            retry_success, retry_result = retry_failed_phase(state, failed_phase)

            # Reset phase_recovered flag after retry attempt
            state.phase_recovered = False

            if retry_success:
                print(f"Phase {failed_phase} succeeded on attempt {state.retry_count}")
                return True, state.retry_count
            else:
                print(f"Retry {state.retry_count} failed for {failed_phase}: {retry_result}")
                error = retry_result  # Update error for next recovery attempt
        else:
            # Recovery failed - append recovery error and continue
            print(f"Recovery failed for {failed_phase} (attempt {state.retry_count}/{PHASE_RETRY_LIMIT}), trying again...")
            error += f" | Recovery error: {recovery_result}"
            continue  # Loop back to increment retry_count and try recovery again

    print(f"Phase {failed_phase} failed after {state.retry_count} retry attempts")
    return False, state.retry_count


def check_for_completion(state: RWLState) -> bool:
    """Checks for a completed.md file."""
    try:
        with open("completed.md", "r") as f:
            return True
    except:
        return False


def run_final_review_cycle(state: RWLState) -> None:
    """Run final REVIEW → BUILD → COMMIT cycle for polishing."""
    # First check the lock
    if not state.lock_token or not check_project_lock(state.lock_token):
        raise Exception('lost the lock; aborting loop')

    print("Running final review cycle...")

    # REVIEW phase
    review_success, review_result = execute_final_review_phase(state)
    if not review_success:
        print("Final review failed, exiting...")
        return

    # BUILD phase (address review concerns rather than task)
    build_success, build_result = execute_final_build_phase(state)
    if not build_success:
        print("Final build failed, exiting...")
        return

    # COMMIT phase
    commit_success, commit_result = execute_commit_phase(state)
    if not commit_success:
        print("Final commit failed, exiting...")
        return

    print("Final review cycle completed successfully!")


def main_loop(state: RWLState) -> None:
    """Main RWL loop orchestrating all phases."""
    save_state_to_disk(state)

    while state.iteration < state.max_iterations and not state.is_complete:
        state.iteration += 1
        # Always run BUILD phase
        success, result = execute_build_phase(state)
        if not success:
            handle_phase_failure(state, Phase.BUILD.value, result)

        state.phase_history.append(f"BUILD_{state.iteration}")

        # Enhanced mode logic
        if state.enhanced_mode:
            # Either follow a REVIEW -> (PLAN/COMMIT) process or fallback to PLAN phase
            if should_trigger_review(state):
                review_success, review_result = execute_review_phase(state)
                state.phase_history.append(f"REVIEW_{state.iteration}")
                if review_success:
                    if Path("review.passed.md").exists():
                        commit_success, commit_result = execute_commit_phase(state)
                        state.phase_history.append(f"COMMIT_{state.iteration}")
                        if not commit_success:
                            handle_phase_failure(state, Phase.COMMIT.value, commit_result)
                    elif Path("review.rejected.md").exists():
                        plan_success, plan_result = execute_plan_phase(state)
                        state.phase_history.append(f"PLAN_{state.iteration}")
                        if not plan_success:
                            handle_phase_failure(state, Phase.PLAN.value, plan_result)
            else:
                plan_success, plan_result = execute_plan_phase(state)
                state.phase_history.append(f"PLAN_{state.iteration}")
                if not plan_success:
                    handle_phase_failure(state, Phase.PLAN.value, plan_result)

        # Classic mode logic
        else:
            plan_success, plan_result = execute_plan_phase(state)
            state.phase_history.append(f"PLAN_{state.iteration}")
            if not plan_success:
                handle_phase_failure(state, Phase.PLAN.value, plan_result)

        # Check for completion promise
        state.is_complete = check_for_completion(state)

        # Save state after each iteration
        save_state_to_disk(state)

    # Check for final review condition
    if state.enhanced_mode and state.iteration < state.max_iterations and state.final_review_requested:
        state.phase_history.append(f"FINAL_REVIEW_CYCLE_{state.iteration}")
        run_final_review_cycle(state)

    # Save final state
    save_state_to_disk(state)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced RWL - AI Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Build a web scraper for news articles"
  %(prog)s --enhanced "Create a REST API with tests"
  %(prog)s --enhanced --final-review "Complex multi-module project"
  %(prog)s --review-every 3 "Regular review cycles"
  %(prog)s --mock-mode "Test without external dependencies"
  %(prog)s --resume "Resume interrupted session"
  %(prog)s --force-resume "Force resume with stale lock"
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}"
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The task description for RWL to work on"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of iterations (default: {DEFAULT_MAX_ITERATIONS})"
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests during development"
    )

    parser.add_argument(
        "--review-every",
        type=int,
        default=DEFAULT_REVIEW_EVERY,
        help="Trigger REVIEW phase every N iterations (default: 0, enhanced mode only)"
    )

    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Enable enhanced four-phase cycle (BUILD-REVIEW-PLAN-COMMIT)"
    )

    parser.add_argument(
        "--final-review",
        action="store_true",
        help="Run final REVIEW → BUILD → COMMIT cycle after completion"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"AI model to use for development (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--review-model",
        default=DEFAULT_REVIEW_MODEL,
        help=f"AI model to use for reviews (default: {DEFAULT_REVIEW_MODEL})"
    )

    parser.add_argument(
        "--push-commits",
        action="store_true",
        help="Push commits to remote repository after successful commit"
    )

    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Use mock mode for testing (no external AI calls)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous session using saved state in .ralph/state.json"
    )

    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume by removing any existing lock file (bypasses staleness check)"
    )

    parser.add_argument(
        "--reorganize-archive",
        action="store_true",
        help="Reorganize archives into per-session directories in .ralph/archive/"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Validate environment
    validate_environment()
    setup_logging()

    # Handle archive reorganization
    if args.reorganize_archive:
        lock_token = None
        lock_file = Path("ralph.lock.json")
        if lock_file.exists():
            try:
                with open(lock_file, "r") as f:
                    lock_data = json.load(f)
                lock_token = lock_data.get("lock_token")
            except Exception:
                pass
        reorganize_archive_files(lock_token)
        return 0

    # Handle resume mode
    if args.resume or args.force_resume:
        loaded_state = load_state_from_disk()
        if not loaded_state:
            print("ERROR: Could not load state. Run without " +
                f"{'--resume' if args.resume else '--force-resume'} .")
            return 1

        lock_file = Path("ralph.lock.json")
        if lock_file.exists():
            if args.force_resume:
                print("WARNING: Removing existing lock file (--force-resume)")
                lock_file.unlink()
            else:
                with open(lock_file, "r") as f:
                    lock_data = json.load(f)
                created_time = lock_data.get("created", 0)
                if time.time() - created_time < STALE_LOCK_TIMEOUT:
                    print("ERROR: Lock file is not stale - another RWL process may be running. "
                          "Use --force-resume to override, or wait for the other process to release the lock.")
                    return 1
                else:
                    print("WARNING: Removing stale lock file")
                    lock_file.unlink()

        print(f"Resuming from iteration {loaded_state.iteration}")
        print(f"Last phase: {loaded_state.current_phase}")
        if loaded_state.last_error:
            print(f"Last error: {loaded_state.last_error}")
        print()

        state = loaded_state
        populate_archived_hashes(state.lock_token) if state.lock_token else ''
        state.lock_token = None
        state.start_time = time.time()
    else:
        state = None

    # Validate prompt
    prompt_content = ""
    if args.prompt:
        prompt_content = args.prompt
        with open("prompt.md", "w") as f:
            f.write(prompt_content)
    elif state and state.original_prompt:
        prompt_content = state.original_prompt
    elif Path("prompt.md").exists():
        with open("prompt.md", "r") as f:
            prompt_content = f.read().strip()
    else:
        print("ERROR: No prompt provided and no prompt.md file found")
        return 1

    # Initialize state if not resuming
    if not state:
        state = RWLState(
            enhanced_mode=args.enhanced,
            skip_tests=args.skip_tests,
            review_every=args.review_every,
            model=args.model,
            review_model=args.review_model,
            max_iterations=args.max_iterations,
            final_review_requested=args.final_review,
            mock_mode=args.mock_mode,
            start_time=time.time(),
            push_commits=args.push_commits,
            original_prompt=prompt_content,
        )

    # Set up signal handlers
    setup_signal_handlers(state)

    # Get project lock
    state.lock_token = get_project_lock()
    if not state.lock_token:
        print("ERROR: Could not acquire project lock")
        return 1

    try:
        # Save initial state
        save_state_to_disk(state)

        # Check if implementation plan exists, create if needed
        if not Path("implementation_plan.md").exists():
            print("No implementation plan found, generating...")
            cli_prompt = args.prompt if args.prompt else ""
            success, result = generate_initial_plan(state, cli_prompt)
            if not success:
                print(f"ERROR: Failed to generate initial plan: {result}")
                return 1

        # Run main loop
        main_loop(state)

        # Check if completed successfully (not by hitting max iterations)
        if state.is_complete and state.iteration < state.max_iterations:
            print("RWL completed successfully!")
            cleanup_process_files(state.lock_token or "")
        else:
            print("RWL hit max iterations and halted.")

        return 0

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        release_project_lock(state.lock_token)


if __name__ == "__main__":
    sys.exit(main())
