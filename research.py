#!/usr/bin/env python3
"""
Research Tool Implementation

A self-contained Python CLI tool that orchestrates AI agent research
through a structured RESEARCH-REVIEW-SYNTHESIZE cycle with breadth-first
search traversal.

Usage:
    python research.py "Your research question"
    python research.py --breadth 5 --depth 2 "Deep research topic"
    python research.py --name "my-research" --resume "Previous research"
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
from string import Template


VERSION = "0.0.1"


DEFAULT_MODEL = "opencode/minimax-m2.1-free"
DEFAULT_REVIEW_MODEL = "opencode/big-pickle"
DEFAULT_BREADTH = 3
DEFAULT_DEPTH = 3
OPENCODE_TIMEOUT = 1200
PHASE_RETRY_LIMIT = 3
RETRY_WAIT_SECONDS = 30
RECOVERY_WAIT_SECONDS = 10
STALE_LOCK_TIMEOUT = 3600


class Phase(Enum):
    RESEARCH = "RESEARCH"
    REVIEW = "REVIEW"
    SYNTHESIZE = "SYNTHESIZE"
    FINAL_REVIEW = "FINAL_REVIEW"
    REVISE = "REVISE"
    RECOVERY = "RECOVERY"


@dataclass
class RWRState:
    """Represents the current state of the RWR process."""
    breadth: int = field(default=DEFAULT_BREADTH)
    depth: int = field(default=DEFAULT_DEPTH)
    iteration: int = field(default=0)
    current_phase: str = field(default=Phase.RESEARCH.value)
    failed_phase: str | None = field(default=None)
    retry_count: int = field(default=0)
    max_iterations: int = field(default=0)
    model: str = field(default=DEFAULT_MODEL)
    review_model: str = field(default=DEFAULT_REVIEW_MODEL)
    lock_token: str | None = field(default=None)
    start_time: float = field(default=0.0)
    phase_history: list[str] = field(default_factory=list)
    last_error: str | None = field(default=None)
    is_complete: bool = field(default=False)
    mock_mode: bool = field(default=False)
    phase_recovered: bool = field(default=False)
    timeout: int = field(default=OPENCODE_TIMEOUT)
    research_name: str = field(default="")
    original_topic: str = field(default="")


def calculate_min_iterations(breadth: int, depth: int) -> int:
    """Calculate minimum safe iteration limit as X^(Y+1) + 5.

    Rationale:
    - A complete breadth-first search with depth Y and breadth X requires approximately X^(Y+1) topics
    - Each topic requires at least one RESEARCH-REVIEW cycle (one iteration)
    - The +5 buffer accounts for edge cases, rejected work requiring re-research, and the final SYNTHESIZE-REVIEW-REVISE cycle
    """
    return breadth ** (depth + 1) + 5


def slugify(text: str) -> str:
    """Convert text to a slug for filename generation."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text)
    return text.strip('-')


def setup_signal_handlers(state: RWRState) -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum: int, frame) -> None:
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        release_project_lock(state.lock_token, state.research_name)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_logging(research_name: str) -> None:
    """Set up logging directory structure."""
    logs_dir = Path(f".research/{research_name}/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)


ARCHIVED_HASHES: set[str] = set()

ARCHIVE_FILENAMES = [
    "state.json",
    "research_plan.md",
    "progress.md",
    "review.accepted.md",
    "review.rejected.md",
    "recovery.notes.md",
    "completed.md",
]

ARCHIVE_FILE_FORMAT = re.compile(r"^(\d+)\.([a-f0-9]{32})\.(.+)$")


def populate_archived_hashes(lock_token: str, research_name: str) -> None:
    """Populate the in-memory hash set from existing archives for this token."""
    archive_dir = Path(f".research/{research_name}/archive")
    if not archive_dir.exists():
        return

    for archive_file in archive_dir.glob(f"*.{lock_token}.*"):
        try:
            content = archive_file.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            ARCHIVED_HASHES.add(file_hash)
        except Exception as e:
            print(f"WARNING: Could not read archive file {archive_file}: {e}")


def archive_intermediate_file(file_path: Path, state: RWRState, prepend: str = "") -> None:
    """Archive an intermediate file with a timestamp prefix if it hasn't been archived yet."""
    if not file_path.exists():
        return

    try:
        content = file_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        if file_hash in ARCHIVED_HASHES:
            return

        archive_dir = Path(f".research/{research_name}/archive")
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        original_name = file_path.name
        if prepend:
            original_name = f"{prepend}.{original_name}"
        archive_name = f"{timestamp}.{lock_token}.{original_name}"
        archive_path = archive_dir / archive_name

        shutil.copy2(file_path, archive_path)
        ARCHIVED_HASHES.add(file_hash)
        print(f"Archived: {file_path.name} -> {archive_name}")
    except Exception as e:
        print(f"WARNING: Could not archive {file_path}: {e}")


def archive_any_process_files(state: RWRState) -> None:
    """Check for and archive any intermediate files."""
    file_dir = Path(f".research/{state.session_name}")
    for file_name in ARCHIVE_FILENAMES:
        archive_intermediate_file(Path(file_dir/file_name), state)

    progress_dir = Path(f".research/{research_name}/progress")
    if progress_dir.exists():
        for progress_file in progress_dir.glob("**/*.md"):
            if progress_file.name != "README.md":
                subtopic = progress_file.parent.name
                archive_intermediate_file(progress_file, lock_token, research_name, prepend=subtopic)


def reorganize_archive_files(lock_token: str | None, research_name: str) -> None:
    """Reorganize archives into per-session directories.

    Skips files matching the current lock token to allow active sessions to continue.
    Only reorganizes sessions with 2 or more files.
    """
    archive_dir = Path(f".research/{research_name}/archive")
    if not archive_dir.exists():
        return

    token_files: dict[str, list[Path]] = {}

    for archive_file in archive_dir.iterdir():
        if not archive_file.is_file():
            continue

        match = ARCHIVE_FILE_FORMAT.match(archive_file.name)
        if match:
            timestamp_str, token, original_name = match.groups()
            if lock_token and token == lock_token:
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


PROMPT_TEMPLATES = {}


def get_template(template_name: str, state: RWRState, default: str) -> str:
    if template_name in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template_name]
        return template if not state.phase_recovered else (template + get_phase_recovery_template())

    template_dir = Path(f'.research/{state.research_name}/templates')
    os.makedirs(template_dir, exist_ok=True)

    file_path = template_dir / f'{template_name}.md'

    if not file_path.exists():
        with open(file_path, 'w') as f:
            f.write(default)
    with open(file_path, 'r') as f:
        template = f.read()
        PROMPT_TEMPLATES[template_name] = template
    template = template or default
    return template if not state.phase_recovered else (template + get_phase_recovery_template())


def get_global_template(template_name: str, default: str) -> str:
    """Get template from global .research/templates directory."""
    template_dir = Path('.research/templates')
    os.makedirs(template_dir, exist_ok=True)

    file_path = template_dir / f'{template_name}.md'

    if not file_path.exists():
        with open(file_path, 'w') as f:
            f.write(default)
    with open(file_path, 'r') as f:
        template = f.read()
    return template or default


def get_phase_recovery_template() -> str:
    if 'phase_recovery' in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES['phase_recovery']

    template_dir = Path('.research/templates')
    os.makedirs(template_dir, exist_ok=True)
    default = '\nThe previous attempt to run this phase failed. Read ' +\
        ' recovery.notes.md for phase recovery information.'

    file_path = template_dir / 'phase_recovery.md'
    if not file_path.exists():
        with open(file_path, 'w') as f:
            f.write(default)
    with open(file_path, 'r') as f:
        template = f.read()
        PROMPT_TEMPLATES['phase_recovery'] = template
    return template or default


def interpolate_template(template: str, state: RWRState, **kwargs) -> str:
    variables = {
        'iteration': state.iteration,
        'current_phase': state.current_phase,
        'failed_phase': state.failed_phase,
        'retry_count': state.retry_count,
        'max_iterations': 'not limited' if state.max_iterations == 0
            else state.max_iterations,
        'breadth': state.breadth,
        'depth': state.depth,
        'model': state.model,
        'review_model': state.review_model,
        'lock_token': state.lock_token,
        'start_time': state.start_time,
        'last_error': state.last_error,
        'is_complete': state.is_complete,
        'mock_mode': state.mock_mode,
        'original_topic': state.original_topic,
        'phase_recovered': state.phase_recovered,
        'research_name': state.research_name,
        **kwargs,
    }
    return Template(template).substitute(variables)


def validate_environment(research_name: str) -> None:
    """Validate the environment before starting."""
    if not Path(".git").exists():
        print("ERROR: This directory is not a git repository")
        sys.exit(1)

    research_dir = Path(f".research/{research_name}")
    try:
        research_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create .research/{research_name} directory: {e}")
        sys.exit(1)


def get_project_lock(research_name: str, force_resume: bool = False) -> str | None:
    """Create a project lock file and return the lock token."""
    research_dir = Path(f".research/{research_name}")
    research_dir.mkdir(parents=True, exist_ok=True)

    lock_file = research_dir / "research.lock.json"

    if lock_file.exists():
        try:
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            created_time = lock_data.get("created", 0)
            if time.time() - created_time < STALE_LOCK_TIMEOUT:
                if force_resume:
                    print("WARNING: Force resuming, removing stale lock file")
                    lock_file.unlink()
                else:
                    print("ERROR: Project is locked by another research instance")
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


def check_project_lock(token: str, research_name: str) -> bool:
    """Check if the project lock is still valid."""
    lock_file = Path(f".research/{research_name}/research.lock.json")

    if not lock_file.exists():
        return False

    try:
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        valid = lock_data.get("lock_token") == token
        if valid:
            lock_data["created"] = time.time()
            with open(lock_file, "w") as f:
                json.dump(lock_data, f)
        return valid
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def release_project_lock(token: str | None, research_name: str) -> None:
    """Release the project lock."""
    if not token:
        return

    lock_file = Path(f".research/{research_name}/research.lock.json")

    try:
        if lock_file.exists():
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            if lock_data.get("lock_token") == token:
                lock_file.unlink()
                print("Project lock released")
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"WARNING: Error releasing lock: {e}")


def cleanup_process_files(lock_token: str, research_name: str) -> None:
    """Clean up lingering process files after successful completion."""
    archive_any_process_files(lock_token, research_name)
    files_to_remove = [
        *ARCHIVE_FILENAMES
    ]

    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            try:
                path.unlink()
                print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"WARNING: Could not remove {file_path}: {e}")

    progress_dir = Path(f".research/{research_name}/progress")
    if progress_dir.exists():
        for progress_file in progress_dir.glob("**/*.md"):
            if progress_file.name != "README.md":
                try:
                    progress_file.unlink()
                    print(f"Cleaned up: {progress_file}")
                except Exception as e:
                    print(f"WARNING: Could not remove {progress_file}: {e}")


def save_state_to_disk(state: RWRState) -> None:
    """Save the current state to disk for crash recovery."""
    state_file = Path(f".research/{state.research_name}/state.json")

    try:
        state_dict = dataclasses.asdict(state)

        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
    except Exception as e:
        print(f"WARNING: Error saving state: {e}")


def load_state_from_disk(research_name: str) -> RWRState | None:
    """Load state from disk for crash recovery."""
    state_file = Path(f".research/{research_name}/state.json")

    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            state_dict = json.load(f)

        if not state_dict.get("original_topic"):
            print("ERROR: Saved state has no original topic. Cannot resume.")
            return None

        return RWRState(**state_dict)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"ERROR: Corrupted state file: {e}")
        return None


def create_research_directory_structure(research_name: str, original_topic: str, breadth: int, depth: int) -> None:
    """Create the .research/{name}/ directory structure with all required subdirectories and files."""
    research_dir = Path(f".research/{research_name}")
    research_dir.mkdir(parents=True, exist_ok=True)

    (research_dir / "archive").mkdir(exist_ok=True)
    (research_dir / "progress").mkdir(exist_ok=True)
    (research_dir / "templates").mkdir(exist_ok=True)
    (research_dir / "logs").mkdir(exist_ok=True)

    reports_dir = Path(f"reports/{research_name}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    lock_data = {
        "lock_token": "",
        "created": 0,
        "pid": 0
    }
    lock_file = research_dir / "research.lock.json"
    with open(lock_file, "w") as f:
        json.dump(lock_data, f)

    state = RWRState(
        original_topic=original_topic,
        breadth=breadth,
        depth=depth,
        max_iterations=calculate_min_iterations(breadth, depth),
        research_name=research_name,
        start_time=time.time()
    )
    save_state_to_disk(state)

    progress_content = f"""# Research Progress

## Summary
- Iteration: 0
- Topics Completed: 0 of 0

## Recent Activity
- Research session initialized for: {original_topic}
"""
    progress_file = research_dir / "progress.md"
    with open(progress_file, "w") as f:
        f.write(progress_content)

    readme_file = research_dir / "progress" / "README.md"
    with open(readme_file, "w") as f:
        f.write("# Progress Directory\n\nPer-topic progress files will be stored here.\n")

    print(f"Created research directory structure: .research/{research_name}/")
    print(f"Created reports directory: reports/{research_name}/")


def call_opencode(
        prompt: str, model: str, lock_token: str, timeout: int = OPENCODE_TIMEOUT,
        mock_mode: bool = False
    ) -> tuple[bool, str]:
    """Call OpenCode with the given prompt and return success status and output."""
    if mock_mode:
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


def generate_initial_plan_prompt(state: RWRState) -> str:
    """Generate the initial research plan prompt."""
    template = get_template('initial_plan', state,
        f"""You are creating an initial research plan.

    TOPIC:
    $original_topic

    PARAMETERS:
    - Breadth: $breadth
    - Depth: $depth

    INSTRUCTIONS:
    1. Analyze the topic to identify root research question(s)
    2. Create depth 0 topics for each major question (up to $breadth topics)
    3. Create research_plan.md with the specified format
    4. Set up acceptance criteria for each topic

    PLAN FORMAT:
    # Research Plan

    ## Metadata
    - Topic: [original research topic or question]
    - Breadth: $breadth
    - Depth: $depth

    ## Topics

    ### [Topic Name] (Depth: 0)
    - Status: Pending
    - Description: Brief description of the topic
    - Acceptance Criteria:
      - Criterion 1
      - Criterion 2
      - Criterion 3

    ## Completion Status
    - Total Topics: N
    - Complete: 0
    - In Progress: 0
    - Pending: N

    OUTPUT: Create research_plan.md with your research plan""")
    return interpolate_template(template, state)


def generate_research_prompt(state: RWRState) -> str:
    """Generate the prompt for the RESEARCH phase."""
    template = get_template('research', state,
        f"""You are a research assistant conducting systematic research.

    TOPIC:
    $original_topic

    PARAMETERS:
    - Breadth: $breadth
    - Depth: $depth
    - Current Iteration: $iteration
    - Max Iterations: $max_iterations

    PHASE: RESEARCH
    Your goal is to conduct research on pending topics using breadth-first search.

    INSTRUCTIONS:
    1. Read research_plan.md to identify topics marked "Pending" or "In Progress"
    2. Prioritize depth 0 topics (high-level questions) first
    3. For each topic:
       - Compile notes from web and file search tools
       - Identify up to $breadth subtopics to explore (if depth < $depth)
       - Write/update progress/[topic_slug].md with detailed findings
       - Update research_plan.md with any new subtopics
       - Mark the topic as "In Review"
    4. Halt after completing research on one topic

    RESEARCH OUTPUT:
    - Update progress/[topic_slug].md with findings, sources, and knowledge gaps
    - Update research_plan.md with status changes and new subtopics

    IMPORTANT:
    - Focus on breadth-first exploration (cover all topics at current depth before going deeper)
    - Do NOT force additional subtopics if fewer than $breadth can be found
    - When depth $depth is reached, no new subtopics should be created
    - Cite sources properly in your findings""")
    return interpolate_template(template, state)


def generate_review_prompt(state: RWRState) -> str:
    """Generate the prompt for the REVIEW phase."""
    template = get_template('review', state,
        f"""You are reviewing research findings for quality and completeness.

    TOPIC:
    $original_topic

    PARAMETERS:
    - Breadth: $breadth
    - Depth: $depth
    - Current Iteration: $iteration

    PHASE: REVIEW
    Your goal is to evaluate research findings against acceptance criteria.

    INSTRUCTIONS:
    1. Read research_plan.md for topics marked "In Review"
    2. Read the corresponding progress/[topic_slug].md file
    3. Perform knowledge gap analysis
    4. Create exactly one of:
       - review.accepted.md: findings meet acceptance criteria
       - review.rejected.md: specific gaps and action items required

    REVIEW CRITERIA:
    - Completeness: Does it address the topic adequately?
    - Quality: Are findings supported by evidence?
    - Sources: Are sources properly cited?
    - Depth: Is the research thorough enough?

    OUTPUT:
    - If ACCEPTED: Create review.accepted.md and mark topic "Complete"
    - If REJECTED: Create review.rejected.md with gaps, mark topic "In Progress"

    CRITICAL: You must create exactly one of review.accepted.md or review.rejected.md.""")
    return interpolate_template(template, state)


def generate_synthesize_prompt(state: RWRState) -> str:
    """Generate the prompt for the SYNTHESIZE phase (final report generation)."""
    template = get_template('synthesize', state,
        f"""You are synthesizing research findings into a comprehensive report.

    TOPIC:
    $original_topic

    PARAMETERS:
    - Breadth: $breadth
    - Depth: $depth
    - Iterations: $iteration

    PHASE: SYNTHESIZE
    Your goal is to compile all research findings into a unified report.

    INSTRUCTIONS:
    1. Read all progress/[topic_slug].md files
    2. Read research_plan.md for topic hierarchy
    3. Create reports/$research_name/report.md with:
       - Executive summary of key findings
       - Findings organized by topic hierarchy
       - Sources/references section with proper citations
       - Clear, professional writing

    REPORT FORMAT:
    # Research Report: [Topic]

    ## Executive Summary
    [Brief overview of key findings]

    ## Introduction
    [Context and research objectives]

    ## Methodology
    [Brief description of research approach]

    ## Findings

    ### [Topic 1]
    [Detailed findings with sources]

    ### [Subtopic 1.1]
    [Subtopic findings with sources]

    ## Sources
    - [Source 1]: [URL or citation]
    - [Source 2]: [URL or citation]

    ## Conclusion
    [Summary and implications]

    OUTPUT: Create reports/$research_name/report.md""")
    return interpolate_template(template, state)


def generate_final_review_prompt(state: RWRState) -> str:
    """Generate the prompt for the FINAL REVIEW phase."""
    template = get_template('final_review', state,
        f"""You are conducting a final quality review of the research report.

    TOPIC:
    $original_topic

    PHASE: FINAL REVIEW
    Your goal is to ensure the report meets quality standards.

    INSTRUCTIONS:
    1. Read the synthesized report reports/$research_name/report.md
    2. Evaluate for:
       - Completeness: Does it address all topics from research_plan.md?
       - Coherence: Is the writing logical and well-structured?
       - Quality: Are findings supported by evidence and properly cited?
       - Professionalism: Is the tone appropriate and free of errors?
    3. Create exactly one of:
       - review.accepted.md: report is satisfactory
       - review.rejected.md: quality issues requiring revision

    OUTPUT:
    - If ACCEPTED: Create review.accepted.md, research is complete
    - If REJECTED: Create review.rejected.md with specific issues, proceed to REVISE phase

    CRITICAL: You must create exactly one of review.accepted.md or review.rejected.md.""")
    return interpolate_template(template, state)


def generate_revise_prompt(state: RWRState) -> str:
    """Generate the prompt for the REVISE phase."""
    template = get_template('revise', state,
        f"""You are revising the research report to address quality issues.

    TOPIC:
    $original_topic

    PHASE: REVISE
    Your goal is to address the quality issues identified in review.rejected.md.

    INSTRUCTIONS:
    1. Read the critique in review.rejected.md
    2. Make targeted revisions to reports/$research_name/report.md
    3. Do NOT add new research or go beyond the original scope
    4. Create review.accepted.md when revisions are complete

    OUTPUT: Update reports/$research_name/report.md with revisions""")
    return interpolate_template(template, state)


def generate_recovery_prompt(state: RWRState) -> str:
    """Generate the prompt for the RECOVERY phase."""
    template = get_template('recovery', state,
        f"""You are diagnosing and recovering from a phase failure.

    TOPIC:
    $original_topic

    PHASE: RECOVERY
    Failed Phase: $failed_phase
    Error: $last_error

    CONTEXT:
    - Current iteration: $iteration
    - Max iterations: $max_iterations
    - Retry count: $retry_count

    INSTRUCTIONS:
    1. Analyze what went wrong in the failed phase
    2. Review .research/$research_name/logs/ for diagnostic information
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
        - Suggest ways to ensure work is broken up enough to avoid another timeout
    - If it was NOT a timeout:
        - Specific steps to resolve the issue
        - Prevention measures for future iterations
        - Whether to retry the failed phase or continue with adjustments

    GOAL: Provide concise, actionable recovery guidance to get research back on track.""")
    return interpolate_template(template, state)


def execute_research_phase(state: RWRState) -> tuple[bool, str]:
    """Execute the RESEARCH phase."""
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    print(f"Starting RESEARCH phase")
    state.current_phase = Phase.RESEARCH.value

    try:
        prompt = generate_research_prompt(state)

        success, result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if success:
            print("RESEARCH phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"RESEARCH phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_review_phase(state: RWRState, is_final: bool = False) -> tuple[bool, str]:
    """Execute the REVIEW phase."""
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    print(f"Starting {'FINAL ' if is_final else ''}REVIEW phase")
    state.current_phase = Phase.FINAL_REVIEW.value if is_final else Phase.REVIEW.value

    try:
        if is_final:
            prompt = generate_final_review_prompt(state)
        else:
            prompt = generate_review_prompt(state)

        success, result = call_opencode(
            prompt, state.review_model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if success:
            if is_final:
                accepted_exists = Path("review.accepted.md").exists()
                rejected_exists = Path("review.rejected.md").exists()

                if accepted_exists ^ rejected_exists:
                    print("FINAL REVIEW phase completed successfully")
                    return True, result
                else:
                    return False, "FINAL REVIEW must create exactly one of: review.accepted.md, review.rejected.md"
            else:
                accepted_exists = Path("review.accepted.md").exists()
                rejected_exists = Path("review.rejected.md").exists()

                if accepted_exists ^ rejected_exists:
                    print("REVIEW phase completed successfully")
                    return True, result
                else:
                    return False, "REVIEW phase must create exactly one of: review.accepted.md, review.rejected.md"
        else:
            return False, result

    except Exception as e:
        error_msg = f"REVIEW phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_synthesize_phase(state: RWRState) -> tuple[bool, str]:
    """Execute the SYNTHESIZE phase."""
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    print("Starting SYNTHESIZE phase")
    state.current_phase = Phase.SYNTHESIZE.value

    try:
        prompt = generate_synthesize_prompt(state)

        success, result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if success:
            print("SYNTHESIZE phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"SYNTHESIZE phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_revise_phase(state: RWRState) -> tuple[bool, str]:
    """Execute the REVISE phase."""
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    print("Starting REVISE phase")
    state.current_phase = Phase.REVISE.value

    try:
        prompt = generate_revise_prompt(state)

        success, result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if success:
            print("REVISE phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"REVISE phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def execute_recovery_phase(state: RWRState) -> tuple[bool, str]:
    """Execute the RECOVERY phase."""
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    print(f"Starting RECOVERY phase for {state.failed_phase}")
    state.current_phase = Phase.RECOVERY.value

    try:
        time.sleep(RECOVERY_WAIT_SECONDS)

        prompt = generate_recovery_prompt(state)

        success, result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if success:
            print("RECOVERY phase completed successfully")
            return True, result
        else:
            return False, result

    except Exception as e:
        error_msg = f"RECOVERY phase error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg


def retry_failed_phase(state: RWRState, failed_phase: str) -> tuple[bool, str]:
    """Retry the specific failed phase."""
    if failed_phase == Phase.RESEARCH.value:
        return execute_research_phase(state)
    elif failed_phase == Phase.REVIEW.value:
        return execute_review_phase(state)
    elif failed_phase == Phase.SYNTHESIZE.value:
        return execute_synthesize_phase(state)
    elif failed_phase == Phase.FINAL_REVIEW.value:
        return execute_review_phase(state, is_final=True)
    elif failed_phase == Phase.REVISE.value:
        return execute_revise_phase(state)
    else:
        return False, f"Invalid phase for retry: {failed_phase}"


def handle_phase_failure(state: RWRState, failed_phase: str, error: str) -> tuple[bool, int]:
    """Handle phase failure with retry logic and recovery."""
    print(f"ERROR: {failed_phase} phase failed: {error}")
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    state.failed_phase = failed_phase
    state.last_error = error

    log_path = Path(f".research/{state.research_name}/logs/errors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{time.time()}: {failed_phase} failed: {error}\n")

    state.retry_count = 0
    recovery_success, recovery_result = False, None

    while state.retry_count < PHASE_RETRY_LIMIT:
        state.retry_count += 1

        if not recovery_success:
            print(f"Attempting RECOVERY phase for {failed_phase} (attempt {state.retry_count}/{PHASE_RETRY_LIMIT})...")
            recovery_success, recovery_result = execute_recovery_phase(state)

        if recovery_success:
            state.phase_recovered = True
            print(f"Recovery successful, retrying {failed_phase} (attempt {state.retry_count}/{PHASE_RETRY_LIMIT})...")

            retry_success, retry_result = retry_failed_phase(state, failed_phase)

            state.phase_recovered = False

            if retry_success:
                print(f"Phase retry {failed_phase} succeeded on attempt {state.retry_count}")
                return True, state.retry_count
            else:
                print(f"Retry {state.retry_count} failed for {failed_phase}: {retry_result}")
                error = retry_result
        else:
            print(f"Recovery failed for {failed_phase} (attempt {state.retry_count}/{PHASE_RETRY_LIMIT}), trying again...")
            error += f" | Recovery error: {recovery_result}"
            continue

    print(f"Phase {failed_phase} failed after {state.retry_count} retry attempts")
    return False, state.retry_count


def check_for_completion(research_name: str) -> bool:
    """Check for the completed.md file."""
    try:
        completed_file = Path(f".research/{research_name}/completed.md")
        if completed_file.exists():
            return True
    except:
        pass
    return False


def check_all_topics_complete(research_name: str) -> bool:
    """Check if all topics in research_plan.md are marked Complete."""
    try:
        plan_file = Path(f".research/{research_name}/research_plan.md")
        if not plan_file.exists():
            return False

        content = plan_file.read_text()
        if "Status: Complete" in content and "Status: In Progress" not in content and "Status: In Review" not in content and "Status: Pending" not in content:
            return True
        if "Status: Complete" in content and content.count("### ") <= content.count("Status: Complete"):
            return True
    except:
        pass
    return False


def get_completion_stats(research_name: str) -> tuple[int, int]:
    """Get completion statistics from research_plan.md."""
    try:
        plan_file = Path(f".research/{research_name}/research_plan.md")
        if not plan_file.exists():
            return 0, 0

        content = plan_file.read_text()
        total = content.count("### ")
        complete = content.count("Status: Complete")

        return complete, total
    except:
        return 0, 0


def run_final_cycle(state: RWRState) -> None:
    """Run SYNTHESIZE → REVIEW → REVISE final cycle."""
    if not state.lock_token or not check_project_lock(state.lock_token, state.research_name):
        raise Exception('lost the lock; aborting loop')

    print("Running final SYNTHESIZE-REVIEW-REVISE cycle...")

    synthesize_success, synthesize_result = execute_synthesize_phase(state)
    if not synthesize_success:
        print("SYNTHESIZE phase failed, exiting...")
        return

    review_success, review_result = execute_review_phase(state, is_final=True)
    if not review_success:
        print("FINAL REVIEW phase failed, exiting...")
        return

    if Path("review.rejected.md").exists():
        revise_success, revise_result = execute_revise_phase(state)
        if not revise_success:
            print("REVISE phase failed, exiting...")
            return

        review_success_2, review_result_2 = execute_review_phase(state, is_final=True)
        if not review_success_2:
            print("FINAL REVIEW phase after revision failed, exiting...")
            return

    print("Final cycle completed successfully!")


def generate_warning_banner(topics_complete: int, topics_total: int, iterations: int, max_iterations: int) -> str:
    """Generate warning banner for incomplete research."""
    return f"""---
⚠️ **WARNING: ITERATION LIMIT REACHED**

This report was generated before all research topics were completed due to reaching the maximum iteration limit. Some findings may be incomplete or missing.

- Topics completed: {topics_complete} of {topics_total}
- Iterations executed: {iterations} (limit: {max_iterations})
- To complete this research, resume with higher --max-iterations or run again without the limit
---
"""


def main_loop(state: RWRState) -> None:
    """Main RWR loop orchestrating research phases."""
    save_state_to_disk(state)
    path_dir = Path(f".research/{state.research_name}")

    while state.iteration < state.max_iterations and not state.is_complete:
        state.iteration += 1

        success, result = execute_research_phase(state)
        if not success:
            handle_phase_failure(state, Phase.RESEARCH.value, result)

        state.phase_history.append(f"RESEARCH_{state.iteration}")
        save_state_to_disk(state)

        if (path_dir/"review.accepted.md").exists():
            # delete review.accepted.md file so it is only used once
            ...
        if (path_dir/"review.rejected.md").exists():
            # delete review.rejected.md file so it is only used once
            ...

        review_success, review_result = execute_review_phase(state)
        state.phase_history.append(f"REVIEW_{state.iteration}")
        if not review_success:
            handle_phase_failure(state, Phase.REVIEW.value, review_result)

        state.is_complete = check_for_completion(state.research_name)
        save_state_to_disk(state)

    complete, total = get_completion_stats(state.research_name)

    if state.iteration >= state.max_iterations and not state.is_complete:
        print(f"WARNING: Iteration limit ({state.max_iterations}) reached before all topics were completed")
        print(f"Topics completed: {complete} of {total}")

    if state.is_complete or state.iteration >= state.max_iterations:
        state.phase_history.append(f"FINAL_CYCLE_{state.iteration}")
        run_final_cycle(state)

        if state.iteration >= state.max_iterations and not state.is_complete:
            report_file = Path(f"reports/{state.research_name}/report.md")
            if report_file.exists():
                banner = generate_warning_banner(complete, total, state.iteration, state.max_iterations)
                content = report_file.read_text()
                report_file.write_text(banner + "\n" + content)
                print("Added iteration limit warning to report")

        cleanup_process_files(state.lock_token or "", state.research_name)

        archive_any_process_files(state.lock_token or "", state.research_name)

        release_project_lock(state.lock_token, state.research_name)

        print("\n" + "="*60)
        print("RESEARCH COMPLETE")
        print("="*60)
        if state.iteration >= state.max_iterations and not state.is_complete:
            print(f"⚠️  Iteration limit reached: {state.iteration}/{state.max_iterations}")
            print(f"Topics completed: {complete} of {total}")
        else:
            print(f"All {total} topics completed in {state.iteration} iterations")
        print(f"Final report: reports/{state.research_name}/report.md")

    save_state_to_disk(state)


def check_template_generation(state: RWRState) -> bool:
    """Validate all prompt templates can be generated without errors."""
    prompt_generators = [
        generate_initial_plan_prompt,
        generate_research_prompt,
        generate_review_prompt,
        generate_synthesize_prompt,
        generate_final_review_prompt,
        generate_revise_prompt,
        generate_recovery_prompt,
    ]

    for generate_fn in prompt_generators:
        try:
            generate_fn(state)
        except KeyError as e:
            template_name = generate_fn.__name__.replace('generate_', '').replace('_prompt', '')
            print(f"ERROR: Template '{template_name}' has invalid variable: {e}")
            return False
        except Exception as e:
            template_name = generate_fn.__name__.replace('generate_', '').replace('_prompt', '')
            print(f"ERROR: Template '{template_name}' failed to generate: {e}")
            return False

    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Research Tool - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Research the history of artificial intelligence"
  %(prog)s --breadth 5 --depth 2 "Deep dive into machine learning"
  %(prog)s --name "ai-history" --resume "Continue previous research"
  %(prog)s --mock-mode "Test without external dependencies"

Note also that you can use a prompt.md file for complex topics."""
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}"
    )

    parser.add_argument(
        "topic",
        nargs="?",
        help="The research topic or question"
    )

    parser.add_argument(
        "--breadth",
        type=int,
        default=DEFAULT_BREADTH,
        help=f"Number of subtopics per topic (default: {DEFAULT_BREADTH})"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Maximum research depth (default: {DEFAULT_DEPTH})"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the research session (used in directory structure)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help=f"Maximum iterations (default: calculated as breadth^(depth+1)+5)"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"AI model to use for research (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--review-model",
        default=DEFAULT_REVIEW_MODEL,
        help=f"AI model to use for reviews (default: {DEFAULT_REVIEW_MODEL})"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=OPENCODE_TIMEOUT,
        help=f"Timeout in seconds for OpenCode calls (default: {OPENCODE_TIMEOUT})"
    )

    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Use mock mode for testing (no external AI calls)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous session"
    )

    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume by removing any existing lock file"
    )

    parser.add_argument(
        "--reorganize-archive",
        action="store_true",
        help="Reorganize archives into per-session directories"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    if args.reorganize_archive:
        lock_token = None
        lock_file = Path(f".research/{args.name}/research.lock.json")
        if lock_file.exists():
            try:
                with open(lock_file, "r") as f:
                    lock_data = json.load(f)
                lock_token = lock_data.get("lock_token")
            except Exception:
                pass
        reorganize_archive_files(lock_token, args.name)
        return 0

    topic = args.topic
    if not topic:
        prompt_file = Path("prompt.md")
        if prompt_file.exists():
            topic = prompt_file.read_text().strip()
        if not topic:
            print("ERROR: No topic provided. Use positional argument or prompt.md file.")
            return 1

    validate_environment(args.name)
    setup_logging(args.name)

    state = None
    if args.resume or args.force_resume:
        state = load_state_from_disk(args.name)
        if not state:
            print(f"ERROR: Could not load state. Run without --resume.")
            return 1

        if args.force_resume:
            lock_file = Path(f".research/{args.name}/research.lock.json")
            if lock_file.exists():
                lock_file.unlink()

        lock_token = get_project_lock(args.name, force_resume=args.force_resume)
        if not lock_token:
            return 1

        state.lock_token = lock_token
        populate_archived_hashes(lock_token, args.name)
    else:
        min_iterations = calculate_min_iterations(args.breadth, args.depth)

        if args.max_iterations > 0 and args.max_iterations < min_iterations:
            print(f"WARNING: Provided max-iterations ({args.max_iterations}) is below minimum ({min_iterations})")
            print(f"Using minimum value: {min_iterations}")
            max_iterations = min_iterations
        elif args.max_iterations > 0:
            max_iterations = args.max_iterations
        else:
            max_iterations = min_iterations

        if max_iterations > 20:
            print(f"WARNING: High iteration count ({max_iterations})")
            if not args.mock_mode:
                print("This may result in extensive research. Continue? (y/n)")
                response = input().lower().strip()
                if response != 'y' and response != 'yes':
                    print("Aborted.")
                    return 1
            else:
                print("Mock mode: continuing without confirmation.")

        create_research_directory_structure(args.name, topic, args.breadth, args.depth)

        lock_token = get_project_lock(args.name)
        if not lock_token:
            return 1

        state = RWRState(
            original_topic=topic,
            breadth=args.breadth,
            depth=args.depth,
            max_iterations=max_iterations,
            model=args.model,
            review_model=args.review_model,
            lock_token=lock_token,
            start_time=time.time(),
            mock_mode=args.mock_mode,
            timeout=args.timeout,
            research_name=args.name,
            iteration=0
        )

        print(f"\n{'='*60}")
        print("RESEARCH TOOL v" + VERSION)
        print(f"{'='*60}")
        print(f"Topic: {topic}")
        print(f"Breadth: {args.breadth}, Depth: {args.depth}")
        print(f"Min iterations required: {min_iterations}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*60}\n")

        setup_signal_handlers(state)

        populate_archived_hashes(lock_token, args.name)

        if not check_template_generation(state):
            print("ERROR: Template validation failed")
            return 1

        print("\nGenerating initial research plan...")

        initial_plan_prompt = generate_initial_plan_prompt(state)
        success, result = call_opencode(
            initial_plan_prompt, state.model, state.lock_token or "",
            timeout=state.timeout, mock_mode=state.mock_mode
        )

        if not success:
            print(f"ERROR: Failed to generate initial plan: {result}")
            return 1

    save_state_to_disk(state)

    main_loop(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
