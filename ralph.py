#!/usr/bin/env python3
"""
Enhanced Ralph Wiggum Implementation

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
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Union


# Configuration Constants
DEFAULT_MODEL = "zen/big-pickle"
DEFAULT_REVIEW_MODEL = "zen/big-pickle"
OPENCODE_TIMEOUT = 600
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_REVIEW_EVERY = 0
PHASE_RETRY_LIMIT = 3
RETRY_WAIT_SECONDS = 30
RECOVERY_WAIT_SECONDS = 10


class Phase(Enum):
    BUILD = "BUILD"
    REVIEW = "REVIEW"
    PLAN = "PLAN"
    COMMIT = "COMMIT"
    RECOVERY = "RECOVERY"


@dataclass
class RalphState:
    """Represents the current state of the Ralph Wiggum process."""
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


def setup_signal_handlers(state: RalphState) -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum: int, frame: Any) -> None:
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        release_project_lock(state.lock_token)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_logging() -> None:
    """Set up logging directory structure."""
    logs_dir = Path(".ralph/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)


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
    lock_file = Path("ralph.lock.md")

    # Check for existing lock
    if lock_file.exists():
        try:
            with open(lock_file, "r") as f:
                content = f.read().strip()

            # Parse existing lock
            lines = content.split('\n')
            token_line = next((line for line in lines if line.startswith("RALPH_LOCK_TOKEN:")), None)
            created_line = next((line for line in lines if line.startswith("CREATED:")), None)

            if token_line and created_line:
                # Check if lock is stale (older than 1 hour)
                try:
                    timestamp_str = created_line.split(": ", 1)[1]
                    created_time = float(timestamp_str)
                    if time.time() - created_time < 3600:  # 1 hour
                        print("ERROR: Project is locked by another Ralph instance")
                        return None
                    else:
                        print("WARNING: Removing stale lock file")
                        lock_file.unlink()
                except ValueError:
                    print("WARNING: Invalid lock file format, removing")
                    lock_file.unlink()
        except Exception as e:
            print(f"WARNING: Error reading lock file: {e}, removing")
            try:
                lock_file.unlink()
            except:
                pass

    # Create new lock
    import secrets
    token = secrets.token_hex(16)
    timestamp = time.time()
    pid = os.getpid()

    lock_content = f"""RALPH_LOCK_TOKEN: {token}
CREATED: {timestamp}
PID: {pid}"""

    try:
        with open(lock_file, "w") as f:
            f.write(lock_content)
        return token
    except Exception as e:
        print(f"ERROR: Cannot create lock file: {e}")
        return None


def check_project_lock(token: str) -> bool:
    """Check if the project lock is still valid."""
    lock_file = Path("ralph.lock.md")

    if not lock_file.exists():
        return False

    try:
        with open(lock_file, "r") as f:
            content = f.read().strip()

        # Check if our token is in the file
        return f"RALPH_LOCK_TOKEN: {token}" in content
    except Exception:
        return False


def release_project_lock(token: str | None) -> None:
    """Release the project lock."""
    if not token:
        return

    lock_file = Path("ralph.lock.md")

    try:
        if lock_file.exists():
            with open(lock_file, "r") as f:
                content = f.read().strip()

            # Only remove if it's our lock
            if f"RALPH_LOCK_TOKEN: {token}" in content:
                lock_file.unlink()
                print("Project lock released")
    except Exception as e:
        print(f"WARNING: Error releasing lock: {e}")


def cleanup_process_files() -> None:
    """Clean up lingering process files after successful completion."""
    files_to_remove = [
        "request.review.md",
        "review.rejected.md",
        "review.passed.md",
        "review.final.md",
        "progress.md",
        ".ralph/recovery.notes.md"
    ]

    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            try:
                path.unlink()
                print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"WARNING: Could not remove {file_path}: {e}")


def save_state_to_disk(state: RalphState) -> None:
    """Save the current state to disk for crash recovery."""
    state_file = Path(".ralph/state.json")

    try:
        # Convert dataclass to dictionary, handling None values properly
        state_dict = dataclasses.asdict(state)

        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
    except Exception as e:
        print(f"WARNING: Error saving state: {e}")


def load_state_from_disk() -> RalphState | None:
    """Load state from disk for crash recovery."""
    state_file = Path(".ralph/state.json")

    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            state_dict = json.load(f)

        # Convert dictionary back to RalphState
        return RalphState(**state_dict)
    except Exception as e:
        print(f"WARNING: Error loading state: {e}")
        return None


def generate_build_meta_prompt(state: RalphState, prompt_content: str, active_task: str|None = None) -> str:
    """Generate the meta-prompt for the BUILD phase."""
    meta_prompt = f"""You are Ralph Wiggum Plus, an AI development assistant working on iteration {state.iteration + 1}.

ORIGINAL PROMPT:
{prompt_content}

PHASE: BUILD
Your goal is to choose a task from the implementation_plan.md file and work toward completing it.

CONTEXT:
- Current iteration: {state.iteration + 1}
- {'You must NOT run tests' if state.skip_tests else 'You should run tests to validate your work'}

INSTRUCTIONS:
1. Read the implementation_plan.md file to identify {'a task that has not been completed' if not active_task else 'the active task: ' + active_task}
2. Implement the task following best practices
3. Write what you learned, what you struggled with, and what remains to be done (if the task was not complete) in progress.md
4. Stage your changes with 'git add'
5. {'Create request.review.md when the task is complete' if state.enhanced_mode else 'Halt'}

TASK SELECTION:
- Choose the highest priority task from implementation_plan.md
- Mark tasks you are working on as "In Progress"
- Mark completed tasks as "In Review" in the implementation_plan.md file
- Focus on one task at a time

PROGRESS TRACKING:
- Update progress.md with: learnings, struggles, remaining work
- If all tasks are marked as "Done" or "Complete" in implementation_plan.md, create completed.md with exactly: <promise>COMPLETE</promise>

IMPORTANT: Focus on quality implementation and clear documentation."""

    return meta_prompt


def generate_review_meta_prompt(state: RalphState, is_final_review: bool = False) -> str:
    """Generate the meta-prompt for the REVIEW phase."""
    if is_final_review:
        meta_prompt = f"""You are conducting a FINAL REVIEW of the completed implementation.

PHASE: FINAL REVIEW
Your goal is to ensure the implementation is polished and ready for delivery.

INSTRUCTIONS:
1. Review all implemented code for quality, consistency, and completeness
2. Check for proper documentation, code style, and best practices
3. Identify any remaining issues or improvements needed
4. Create review.final.md with your findings

REVIEW CRITERIA:
- Code quality and maintainability
- Documentation completeness
- Test coverage (unless tests were skipped)
- Adherence to project requirements
- Overall polish and professionalism

OUTPUT:
- If everything is satisfactory: create review.passed.md
- If improvements are needed: create review.rejected.md with specific action items"""
    else:
        meta_prompt = f"""You are reviewing the completion of a task in the development cycle.

PHASE: REVIEW
Your goal is to evaluate if the completed task meets quality standards.

INSTRUCTIONS:
1. Read request.review.md to understand what was completed
2. Review the implementation changes (check git status, diff staged files)
3. Evaluate against project requirements and best practices
4. Create either review.passed.md OR review.rejected.md

REVIEW CRITERIA:
- Task completion: Is the task fully implemented?
- Code quality: Does it follow best practices?
- Testing: Are appropriate tests included (unless --skip-tests)?
- Documentation: Is the code properly documented?

OUTPUT FORMAT:
- If PASSED: Create review.passed.md with approval and any minor suggestions
- If REJECTED: Create review.rejected.md with specific issues and action items

CRITICAL: You must create exactly one of these files."""

    return meta_prompt


def generate_plan_meta_prompt(state: RalphState) -> str:
    """Generate the meta-prompt for the PLAN phase."""
    meta_prompt = f"""You are planning the next development tasks.

PHASE: PLAN
Your goal is to update the implementation plan based on current progress and feedback.

CONTEXT:
- Current iteration: {state.iteration + 1}
- Enhanced mode: {'Yes' if state.enhanced_mode else 'No'}

INSTRUCTIONS:
1. Read review.rejected.md if it exists (for feedback on rejected work)
2. Read progress.md if it exists (for learnings and struggles)
3. Read the current implementation_plan.md
4. Update implementation_plan.md with prioritized tasks

TASK PRIORITIZATION:
- Address any issues from review.rejected.md first
- Focus on remaining implementation work
- Consider dependencies between tasks
- Prioritize tasks that unblock other work

PLAN FORMAT:
- Use clear, actionable task descriptions
- Mark tasks as priority: [HIGH], [MEDIUM], [LOW]
- Include acceptance criteria for complex tasks
- Keep tasks focused and achievable

OUTPUT: Update implementation_plan.md with the revised plan"""

    return meta_prompt


def generate_commit_meta_prompt(state: RalphState) -> str:
    """Generate the meta-prompt for the COMMIT phase."""
    meta_prompt = f"""You are preparing to commit completed work.

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

    return meta_prompt


def generate_recovery_meta_prompt(state: RalphState, failed_phase: str, error: str) -> str:
    """Generate the meta-prompt for the RECOVERY phase."""
    meta_prompt = f"""You are diagnosing and recovering from a phase failure.

PHASE: RECOVERY
Failed Phase: {failed_phase}
Error: {error}

CONTEXT:
- Current iteration: {state.iteration + 1}
- Retry count: {state.retry_count}
- Enhanced mode: {'Yes' if state.enhanced_mode else 'No'}

INSTRUCTIONS:
1. Analyze what went wrong in the failed phase
2. Review .ralph/state.json and .ralph/logs/ for diagnostic information
3. Identify the root cause of the failure
4. Create a recovery plan in .ralph/recovery.notes.md

DIAGNOSTIC STEPS:
- Check for file system issues (permissions, disk space)
- Verify git repository state
- Look for configuration or environment problems
- Check for timeout or network issues

RECOVERY OUTPUT (.ralph/recovery.notes.md):
- Root cause analysis
- Specific steps to resolve the issue
- Prevention measures for future iterations
- Whether to retry the failed phase or continue with adjustments

GOAL: Provide actionable recovery guidance to get development back on track."""

    return meta_prompt


def call_opencode(prompt: str, model: str, timeout: int = OPENCODE_TIMEOUT, mock_mode: bool = False) -> Tuple[bool, str]:
    """Call OpenCode with the given prompt and return success status and output."""
    if mock_mode:
        # Mock mode for testing
        return True, f"MOCK RESPONSE for: {prompt[:100]}..."

    try:
        cmd = ["opencode", "run", "--model", model, prompt]

        print(f"Executing: opencode --model {model}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd()
        )

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


def generate_initial_plan(state: RalphState, prompt_content: str) -> Tuple[bool, str]:
    """Generate the initial implementation plan when none exists."""
    initial_plan_prompt = f"""You are the Ralph Wiggum Plus AI coding assistant creating an initial implementation plan.

ORIGINAL PROMPT:
{prompt_content}

INSTRUCTIONS:
1. Analyze the task and break it down into concrete implementation steps
2. Create implementation_plan.md with prioritized tasks
3. Focus on a logical progression from foundation to completion

TASK PLANNING:
- Break the task into manageable sub-tasks
- Identify dependencies between tasks
- Prioritize tasks that enable subsequent work
- Consider testing, documentation, and polish tasks

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
    plan_review_prompt = f"""You are Ralph Wiggum Plus AI coding assistant. Your
task is to review an implementation plan.
ORIGINAL PROMPT
{prompt_content}

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
updates, skip this step.
"""

    revise_plan_prompt = f"""You are Ralph Wiggum Plus AI coding assistant. Your
task is to revise an implementation plan to incorporate and address the critique
from the reviewer.

ORIGINAL PROMPT:
{prompt_content}

INSTRUCTIONS:
1. Read the plan written to implementation_plan.md.
2. Read the analysis/critique provided in plan.review.md.
3. Consider carefully what parts of the analysis/critique are valid and which are not.
4. If there were any valid points to the analysis/critique, update implementation_plan.md
to address those concerns/incorporate that feedback.
5. Delete the review.plan.md file when you are done.
"""

    # Execute the plan generation
    success, result = call_opencode(
        initial_plan_prompt, state.model, mock_mode=state.mock_mode
    )
    total_result = result
    if not success:
        return False, result

    # Now enter plan revision loop
    for _ in range(2):
        success, result = call_opencode(
            plan_review_prompt, state.review_model, mock_mode=state.mock_mode
        )
        total_result += f'\n\n{result}'
        if not success:
            return False, result
        if not Path('plan.review.md').exists():
            return True, result
        success, result = call_opencode(
            revise_plan_prompt, state.model, mock_mode=state.mock_mode
        )
        total_result += f'\n\n{result}'
        if not success:
            return False, total_result

    return True, total_result

def execute_build_phase(state: RalphState) -> Tuple[bool, str]:
    """Execute the BUILD phase."""
    print(f"Starting BUILD phase (iteration {state.iteration + 1})")

    try:
        # Read the prompt and implementation plan
        prompt_content = ""
        if Path("prompt.md").exists():
            with open("prompt.md", "r") as f:
                prompt_content = f.read().strip()

        # Generate build meta-prompt
        meta_prompt = generate_build_meta_prompt(state, prompt_content)

        # Execute via OpenCode
        success, result = call_opencode(meta_prompt, state.model, mock_mode=state.mock_mode)

        if success:
            print("BUILD phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"BUILD phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_review_phase(state: RalphState, is_final_review: bool = False) -> Tuple[bool, str]:
    """Execute the REVIEW phase."""
    print("Starting REVIEW phase")

    try:
        # Wait for request.review.md in enhanced mode (for regular reviews)
        if not is_final_review and state.enhanced_mode:
            request_file = Path("request.review.md")
            if not request_file.exists():
                print(f"Waiting {RECOVERY_WAIT_SECONDS} seconds for request.review.md...")
                time.sleep(RECOVERY_WAIT_SECONDS)
                if not request_file.exists():
                    return False, "request.review.md file not found"

            # Read and delete request.review.md
            with open(request_file, "r") as f:
                request_content = f.read()
            request_file.unlink()

        # Generate review meta-prompt
        meta_prompt = generate_review_meta_prompt(state, is_final_review)

        # Execute via OpenCode
        success, result = call_opencode(meta_prompt, state.review_model, mock_mode=state.mock_mode)

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


def execute_plan_phase(state: RalphState) -> Tuple[bool, str]:
    """Execute the PLAN phase."""
    print("Starting PLAN phase")

    try:
        # Read feedback files if they exist
        review_rejected_content = ""
        progress_content = ""

        if Path("review.rejected.md").exists():
            with open("review.rejected.md", "r") as f:
                review_rejected_content = f.read()

        if Path("progress.md").exists():
            with open("progress.md", "r") as f:
                progress_content = f.read()

        # Generate plan meta-prompt
        meta_prompt = generate_plan_meta_prompt(state)

        # Execute via OpenCode
        success, result = call_opencode(meta_prompt, state.model, mock_mode=state.mock_mode)

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


def execute_commit_phase(state: RalphState) -> Tuple[bool, str]:
    """Execute the COMMIT phase."""
    print("Starting COMMIT phase")

    try:
        # Read review.passed.md
        review_passed_content = ""
        if Path("review.passed.md").exists():
            with open("review.passed.md", "r") as f:
                review_passed_content = f.read()
        else:
            return False, "review.passed.md not found"

        # Generate commit meta-prompt
        meta_prompt = generate_commit_meta_prompt(state)

        # Execute via OpenCode
        success, result = call_opencode(meta_prompt, state.model, mock_mode=state.mock_mode)

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


def execute_recovery_phase(state: RalphState, failed_phase: str, error: str) -> Tuple[bool, str]:
    """Execute the RECOVERY phase."""
    print(f"Starting RECOVERY phase for {failed_phase}")

    try:
        # Wait briefly for transient issues to resolve
        time.sleep(RECOVERY_WAIT_SECONDS)

        # Generate recovery meta-prompt
        meta_prompt = generate_recovery_meta_prompt(state, failed_phase, error)

        # Execute via OpenCode
        success, result = call_opencode(meta_prompt, state.model, mock_mode=state.mock_mode)

        if success:
            print("RECOVERY phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"RECOVERY phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_final_review_phase(state: RalphState) -> Tuple[bool, str]:
    """Execute the final REVIEW phase."""
    return execute_review_phase(state, is_final_review=True)


def should_trigger_review(state: RalphState) -> bool:
    """Determine if REVIEW phase should be triggered."""
    # Enhanced mode: check for request.review.md file
    if state.enhanced_mode and Path("request.review.md").exists():
        return True

    # Check review-every parameter
    if state.review_every > 0 and state.iteration > 0:
        return state.iteration % state.review_every == 0

    return False


def handle_phase_failure(state: RalphState, failed_phase: str, error: str) -> RalphState:
    """Handle phase failure with retry logic and recovery."""
    print(f"ERROR: {failed_phase} phase failed: {error}")
    state.failed_phase = failed_phase
    state.last_error = error
    state.retry_count += 1

    # Log the failure
    log_path = Path(f".ralph/logs/errors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{time.time()}: {failed_phase} failed: {error}\n")

    # Enhanced mode: try recovery if under retry limit
    if state.enhanced_mode and state.retry_count < PHASE_RETRY_LIMIT:
        print(f"Attempting RECOVERY phase for {failed_phase}...")
        recovery_success, recovery_result = execute_recovery_phase(state, failed_phase, error)
        if recovery_success:
            print(f"Recovery successful, retrying {failed_phase}...")
            state.retry_count = 0  # Reset after successful recovery
        else:
            print(f"Recovery failed, continuing with current state...")
    else:
        print(f"Maximum retries reached for {failed_phase}, continuing...")

    return state


def check_for_completion(state: RalphState) -> bool:
    """Checks for a completed.md file."""
    try:
        with open("completed.md", "r") as f:
            return "<promise>COMPLETE</promise>" in f.read()
    except:
        return False


def run_final_review_cycle(state: RalphState) -> None:
    """Run final REVIEW → BUILD → COMMIT cycle for polishing."""
    if not state.enhanced_mode:
        return

    print("Running final review cycle...")

    # REVIEW phase
    review_success, review_result = execute_final_review_phase(state)
    if not review_success:
        print("Final review failed, exiting...")
        return

    # BUILD phase (address review concerns rather than task)
    build_success, build_result = execute_build_phase(state)
    if not build_success:
        print("Final build failed, exiting...")
        return

    # COMMIT phase
    commit_success, commit_result = execute_commit_phase(state)
    if not commit_success:
        print("Final commit failed, exiting...")
        return

    print("Final review cycle completed successfully!")


def main_loop(state: RalphState) -> None:
    """Main Ralph loop orchestrating all phases."""
    save_state_to_disk(state)

    while state.iteration < state.max_iterations and not state.is_complete:
        # Always run BUILD phase
        success, result = execute_build_phase(state)
        if not success:
            state = handle_phase_failure(state, Phase.BUILD.value, result)
            continue

        state.iteration += 1
        state.phase_history.append(f"BUILD_{state.iteration}")

        # Enhanced mode logic
        if state.enhanced_mode:
            # Either follow a review process or fallback to PLAN phase
            if should_trigger_review(state):
                review_success, review_result = execute_review_phase(state)
                if review_success:
                    if Path("review.passed.md").exists():
                        commit_success, commit_result = execute_commit_phase(state)
                        if not commit_success:
                            state = handle_phase_failure(state, Phase.COMMIT.value, commit_result)
                    elif Path("review.rejected.md").exists():
                        plan_success, plan_result = execute_plan_phase(state)
                        if not plan_success:
                            state = handle_phase_failure(state, Phase.PLAN.value, plan_result)
            else:
                plan_success, plan_result = execute_plan_phase(state)
                if not plan_success:
                    state = handle_phase_failure(state, Phase.PLAN.value, plan_result)

        # Classic mode logic
        else:
            plan_success, plan_result = execute_plan_phase(state)
            if not plan_success:
                state = handle_phase_failure(state, Phase.PLAN.value, plan_result)

        # Check for completion promise
        state.is_complete = check_for_completion(state)

        # Save state after each iteration
        save_state_to_disk(state)

    # Check for final review condition
    if state.enhanced_mode and state.iteration < state.max_iterations and state.final_review_requested:
        run_final_review_cycle(state)

    # Save final state
    save_state_to_disk(state)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Ralph Wiggum - AI Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Build a web scraper for news articles"
  %(prog)s --enhanced "Create a REST API with tests"
  %(prog)s --enhanced --final-review "Complex multi-module project"
  %(prog)s --review-every 3 "Regular review cycles"
  %(prog)s --mock-mode "Test without external dependencies"
        """
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The task description for Ralph to work on"
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

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Validate prompt
    prompt_content = ""
    if args.prompt:
        prompt_content = args.prompt
        # Save prompt to prompt.md
        with open("prompt.md", "w") as f:
            f.write(prompt_content)
    elif Path("prompt.md").exists():
        with open("prompt.md", "r") as f:
            prompt_content = f.read().strip()
    else:
        print("ERROR: No prompt provided and no prompt.md file found")
        return 1

    # Validate environment
    validate_environment()
    setup_logging()

    # Initialize state
    state = RalphState(
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
            success, result = generate_initial_plan(state, prompt_content)
            if not success:
                print(f"ERROR: Failed to generate initial plan: {result}")
                return 1

        # Run main loop
        main_loop(state)

        print("Ralph completed successfully!")

        # Check if completed successfully (not by hitting max iterations)
        if state.is_complete and state.iteration < state.max_iterations:
            cleanup_process_files()

        return 0

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        release_project_lock(state.lock_token)


if __name__ == "__main__":
    sys.exit(main())
