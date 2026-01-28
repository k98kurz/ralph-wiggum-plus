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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from string import Template
from time import time, sleep
from typing import Any, Callable, Generic, TypeVar
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

# semver string
VERSION = "0.0.1-dev"


# Configuration Constants
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "opencode/big-pickle")
DEFAULT_REVIEW_MODEL = os.getenv("DEFAULT_REVIEW_MODEL", "opencode/big-pickle")
DEFAULT_BREADTH = int(os.getenv("DEFAULT_BREADTH", 3))
DEFAULT_DEPTH = int(os.getenv("DEFAULT_DEPTH", 3))
OPENCODE_TIMEOUT = int(os.getenv("OPENCODE_TIMEOUT", 1200))
PHASE_RETRY_LIMIT = int(os.getenv("PHASE_RETRY_LIMIT", 3))
RETRY_WAIT_SECONDS = int(os.getenv("RETRY_WAIT_SECONDS", 30))
RECOVERY_WAIT_SECONDS = int(os.getenv("RECOVERY_WAIT_SECONDS", 10))
STALE_LOCK_TIMEOUT = int(os.getenv("STALE_LOCK_TIMEOUT", 3600))

# Directory Constants
RESEARCH_DIR = ".research"

# Filename Constants
STATE_FILE = "state.json"
LOCK_FILE = "research.lock.json"
RESEARCH_PLAN_FILE = "research_plan.md"
PROGRESS_FILE = "progress.md"
REVIEW_ACCEPTED_FILE = "review.accepted.md"
REVIEW_REJECTED_FILE = "review.rejected.md"
RECOVERY_NOTES_FILE = "recovery.notes.md"
COMPLETED_FILE = "completed.md"

# Subdirectory Constants
ARCHIVE_DIR = "archive"
LOGS_DIR = "logs"
TEMPLATES_DIR = "templates"
PROGRESS_DIR = "progress"
REPORTS_DIR = "reports"


# functional Result type stuff
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class Ok(Generic[T]):
    data: T
    success: bool = True
    def map(self, fn: Callable[[T], U]) -> Result[U]:
        return try_except(fn, self.data)

@dataclass
class Err:
    error: Exception
    success: bool = False
    def map(self, fn: Callable[..., object]) -> Err:
        return self

Result = Ok[T] | Err

def success(data: T) -> Ok[T]:
    return Ok(data)

def failure(error: Exception) -> Err:
    return Err(error)

def try_except(fn: Callable[[...], T], *args, **kwargs) -> Result[T]: # type: ignore
    try:
        return success(fn(*args, **kwargs))
    except Exception as e:
        return failure(e) # type: ignore

def pipe(
        initial: Any,
        *fns: Callable[[Result[T]], Result[T]]
    ) -> Result[T]:
    result = initial if isinstance(initial, (Ok, Err)) else success(initial)
    for fn in fns:
        if result.success:
            result = fn(result)
    return result


# formatting helper function
def lstrip_lines(text: str) -> str:
    return "\n".join([l.lstrip() for l in text.split("\n")])


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
        A complete breadth-first search with depth Y and breadth X
        requires approximately X^(Y+1) topics; each topic requires at
        least one RESEARCH-REVIEW cycle (oneiteration); the +5 buffer
        accounts for edge cases, rejected work requiring re-research,
        and the final SYNTHESIZE-REVIEW-REVISE cycle
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
        release_project_lock(state)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_logging(research_name: str) -> None:
    """Set up logging directory structure."""
    logs_dir = get_research_path(research_name, LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)


# Archive Constants
ARCHIVED_HASHES: set[str] = set()

ARCHIVE_FILENAMES = [
    STATE_FILE,
    RESEARCH_PLAN_FILE,
    PROGRESS_FILE,
    REVIEW_ACCEPTED_FILE,
    REVIEW_REJECTED_FILE,
    RECOVERY_NOTES_FILE,
    COMPLETED_FILE,
]

ARCHIVE_FILE_FORMAT = re.compile(r"^(\d+)\.([a-f0-9]{32})\.(.+)$")


def get_research_path(research_name: str, *path_parts: str) -> Path:
    """Generate per-research path preserving existing directory structure."""
    return Path(f"{RESEARCH_DIR}/{research_name}", *path_parts)


def populate_archived_hashes(state: RWRState) -> None:
    """Populate the in-memory hash set from existing archives for this token."""
    archive_dir = get_research_path(state.research_name, ARCHIVE_DIR)
    if not archive_dir.exists():
        return

    for archive_file in archive_dir.glob(f"*.{state.lock_token}.*"):
        try:
            content = archive_file.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            ARCHIVED_HASHES.add(file_hash)
        except Exception as e:
            print(f"WARNING: Could not read archive file {archive_file}: {e}")


def archive_intermediate_file(
        file_path: Path, state: RWRState, prepend: str = ""
    ) -> None:
    """Archive an intermediate file with a timestamp prefix if it hasn't
        been archived yet.
    """
    if not file_path.exists():
        return

    try:
        content = file_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        if file_hash in ARCHIVED_HASHES:
            return

        archive_dir = get_research_path(state.research_name, ARCHIVE_DIR)
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time())
        original_name = file_path.name
        if prepend:
            original_name = f"{prepend}.{original_name}"
        archive_name = f"{timestamp}.{state.lock_token}.{original_name}"
        archive_path = archive_dir / archive_name

        shutil.copy2(file_path, archive_path)
        ARCHIVED_HASHES.add(file_hash)
        print(f"Archived: {file_path.name} -> {archive_name}")
    except Exception as e:
        print(f"WARNING: Could not archive {file_path}: {e}")


def archive_any_process_files(state: RWRState) -> None:
    """Check for and archive any intermediate files."""
    file_dir = get_research_path(state.research_name)
    for file_name in ARCHIVE_FILENAMES:
        archive_intermediate_file(file_dir / file_name, state)

    progress_dir = get_research_path(state.research_name, PROGRESS_DIR)
    if progress_dir.exists():
        for progress_file in progress_dir.glob("**/*.md"):
            if progress_file.name != "README.md":
                subtopic = progress_file.parent.name
                archive_intermediate_file(progress_file, state, prepend=subtopic)


def reorganize_archive_files(
        state: RWRState, current_token: str | None = None
    ) -> None:
    """Reorganize archives into per-session directories.

    Skips files matching the current lock token to allow active sessions to continue.
    Only reorganizes sessions with 2 or more files.
    """
    archive_dir = get_research_path(state.research_name, ARCHIVE_DIR)
    if not archive_dir.exists():
        return

    token_files: dict[str, list[Path]] = {}

    for archive_file in archive_dir.iterdir():
        if not archive_file.is_file():
            continue

        match = ARCHIVE_FILE_FORMAT.match(archive_file.name)
        if match:
            timestamp_str, token, original_name = match.groups()
            if current_token is not None:
                token_to_skip = current_token
            else:
                token_to_skip = state.lock_token
            if token_to_skip and token == token_to_skip:
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
        if not state.phase_recovered:
            return template
        return (template + get_phase_recovery_template())

    template_dir = get_research_path(state.research_name, TEMPLATES_DIR)
    os.makedirs(template_dir, exist_ok=True)

    file_path = template_dir / f'{template_name}.md'

    if not file_path.exists():
        with open(file_path, 'w') as f:
            f.write(default)
    with open(file_path, 'r') as f:
        template = f.read()
        PROMPT_TEMPLATES[template_name] = template
    template = template or default
    if not state.phase_recovered:
        return template
    return (template + get_phase_recovery_template())


def get_global_template(template_name: str, default: str) -> str:
    """Get template from global .research/templates directory."""
    template_dir = Path(RESEARCH_DIR) / TEMPLATES_DIR
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

    template_dir = Path(RESEARCH_DIR) / TEMPLATES_DIR
    os.makedirs(template_dir, exist_ok=True)
    default = '\nThe previous attempt to run this phase failed. Read ' +\
        f' {RECOVERY_NOTES_FILE} for phase recovery information.'

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

    research_dir = get_research_path(research_name)
    try:
        research_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create {get_research_path(research_name)} directory: {e}")
        sys.exit(1)


def get_project_lock(research_name: str, force_resume: bool = False) -> str | None:
    """Create a project lock file and return the lock token."""
    research_dir = get_research_path(research_name)
    research_dir.mkdir(parents=True, exist_ok=True)

    lock_file = research_dir / LOCK_FILE

    if lock_file.exists():
        try:
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            created_time = lock_data.get("created", 0)
            if time() - created_time < STALE_LOCK_TIMEOUT:
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
    timestamp = time()
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


def check_project_lock(state: RWRState) -> bool:
    """Check if the project lock is still valid."""
    lock_file = get_research_path(state.research_name, LOCK_FILE)

    if not lock_file.exists():
        return False

    try:
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        valid = lock_data.get("lock_token") == state.lock_token
        if valid:
            lock_data["created"] = time()
            with open(lock_file, "w") as f:
                json.dump(lock_data, f)
        return valid
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def release_project_lock(state: RWRState) -> None:
    """Release the project lock."""
    if not state.lock_token:
        return

    lock_file = get_research_path(state.research_name, LOCK_FILE)

    try:
        if lock_file.exists():
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            if lock_data.get("lock_token") == state.lock_token:
                lock_file.unlink()
                print("Project lock released")
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"WARNING: Error releasing lock: {e}")


def cleanup_process_files(state: RWRState) -> None:
    """Clean up lingering process files after successful completion."""
    archive_any_process_files(state)
    files_to_remove = [
        *ARCHIVE_FILENAMES
    ]

    research_dir = get_research_path(state.research_name)
    for file_path in files_to_remove:
        path = research_dir / file_path
        if path.exists():
            try:
                path.unlink()
                print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"WARNING: Could not remove {file_path}: {e}")

    progress_dir = research_dir / "progress"
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
    state_file = get_research_path(state.research_name, STATE_FILE)

    try:
        state_dict = dataclasses.asdict(state)

        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
    except Exception as e:
        print(f"WARNING: Error saving state: {e}")


def load_state_from_disk(research_name: str) -> Result[RWRState]:
    """Load state from disk for crash recovery."""
    state_file = get_research_path(research_name, STATE_FILE)

    if not state_file.exists():
        return failure(Exception(f"state file {STATE_FILE} not found"))

    with open(state_file, "r") as f:
        result = try_except(json.load, f)
        if not result.success:
            print(f"ERROR: failed to load state file: {result.error}")
            return result

        if "original_topic" not in result.data:
            print("ERROR: Saved state has no original topic. Cannot resume.")
            return failure(
                Exception("Saved state has no original topic. Cannot resume.")
            )

        return success(RWRState(**result.data))


def create_research_directory_structure(
        research_name: str, original_topic: str, breadth: int, depth: int
    ) -> None:
    """Create the .research/{name}/ directory structure with all required
        subdirectories and files.
    """
    research_dir = get_research_path(research_name)
    research_dir.mkdir(parents=True, exist_ok=True)

    (research_dir / ARCHIVE_DIR).mkdir(exist_ok=True)
    (research_dir / PROGRESS_DIR).mkdir(exist_ok=True)
    (research_dir / TEMPLATES_DIR).mkdir(exist_ok=True)
    (research_dir / LOGS_DIR).mkdir(exist_ok=True)

    reports_dir = Path(REPORTS_DIR) / research_name
    reports_dir.mkdir(parents=True, exist_ok=True)

    lock_data = {
        "lock_token": "",
        "created": 0,
        "pid": 0
    }
    lock_file = research_dir / LOCK_FILE
    with open(lock_file, "w") as f:
        json.dump(lock_data, f)

    state = RWRState(
        original_topic=original_topic,
        breadth=breadth,
        depth=depth,
        max_iterations=calculate_min_iterations(breadth, depth),
        research_name=research_name,
        start_time=time()
    )
    save_state_to_disk(state)

    progress_content = f"""# Research Progress

    ## Summary
    - Iteration: 0
    - Topics Completed: 0 of 0

    ## Recent Activity
    - Research session initialized for: {original_topic}"""
    progress_file = research_dir / PROGRESS_FILE
    with open(progress_file, "w") as f:
        f.write(progress_content)

    readme_file = research_dir / "progress" / "README.md"
    with open(readme_file, "w") as f:
        f.write("# Progress Directory\n\nPer-topic progress files will be stored here.\n")

    print(f"Created research directory structure: {get_research_path(research_name)}/")
    print(f"Created reports directory: {Path(REPORTS_DIR) / research_name}/")


def call_opencode(
        prompt: str, model: str, lock_token: str, timeout: int = OPENCODE_TIMEOUT,
        mock_mode: bool = False
    ) -> Result[str]:
    """Call OpenCode with the given prompt and return success status and output."""
    if mock_mode:
        return success(f"MOCK RESPONSE for: {prompt[:100]}...")

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
            return success(result.stdout)
        else:
            error_msg = f"OpenCode failed with code {result.returncode}: {result.stderr}"
            print(f"ERROR: {error_msg}")
            return failure(Exception(error_msg))

    except subprocess.TimeoutExpired:
        error_msg = f"OpenCode timed out after {timeout} seconds"
        print(f"ERROR: {error_msg}")
        return failure(TimeoutError(error_msg))
    except Exception as e:
        error_msg = f"Error calling OpenCode: {e}"
        print(f"ERROR: {error_msg}")
        return failure(Exception(error_msg))


def generate_initial_plan_prompt(state: RWRState) -> str:
    """Generate the initial research plan prompt."""
    template = get_template('initial_plan', state, lstrip_lines(
        f"""You are creating an initial research plan.

    TOPIC:
    $original_topic

    PARAMETERS:
    - Breadth: $breadth
    - Depth: $depth

    INSTRUCTIONS:
    1. Analyze the topic to identify root research question(s)
    2. Create depth 0 topics for each major question (up to $breadth topics)
    3. Create {RESEARCH_PLAN_FILE} with the specified format
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

    OUTPUT: Create {RESEARCH_PLAN_FILE} with your research plan"""))
    return interpolate_template(template, state)


def generate_research_prompt(state: RWRState) -> str:
    """Generate the prompt for the RESEARCH phase."""
    template = get_template('research', state, lstrip_lines(
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
    1. Read {RESEARCH_PLAN_FILE} to identify topics marked "Pending" or "In Progress"
    2. Prioritize depth 0 topics (high-level questions) first
    3. For each topic:
       - Compile notes from web and file search tools
       - Identify up to $breadth subtopics to explore (if depth < $depth)
       - Write/update progress/[topic_slug].md with detailed findings
       - Update {RESEARCH_PLAN_FILE} with any new subtopics
       - Mark the topic as "In Review"
    4. Halt after completing research on one topic

    RESEARCH OUTPUT:
    - Update progress/[topic_slug].md with findings, sources, and knowledge gaps
    - Update {RESEARCH_PLAN_FILE} with status changes and new subtopics

    IMPORTANT:
    - Focus on breadth-first exploration (cover all topics at current depth
    before going deeper)
    - Do NOT force additional subtopics if fewer than $breadth can be found
    - When depth $depth is reached, no new subtopics should be created
    - Cite sources properly in your findings"""))
    return interpolate_template(template, state)


def generate_review_prompt(state: RWRState) -> str:
    """Generate the prompt for the REVIEW phase."""
    template = get_template('review', state, lstrip_lines(
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
    1. Read {RESEARCH_PLAN_FILE} for topics marked "In Review"
    2. Read the corresponding progress/[topic_slug].md file
    3. Perform knowledge gap analysis
    4. Create exactly one of:
       - {REVIEW_ACCEPTED_FILE}: findings meet acceptance criteria
       - {REVIEW_REJECTED_FILE}: specific gaps and action items required

    REVIEW CRITERIA:
    - Completeness: Does it address the topic adequately?
    - Quality: Are findings supported by evidence?
    - Sources: Are sources properly cited?
    - Depth: Is the research thorough enough?

    OUTPUT:
    - If ACCEPTED: Create {REVIEW_ACCEPTED_FILE} and mark topic "Complete"
    - If REJECTED: Create {REVIEW_REJECTED_FILE} with gaps, mark topic "In Progress"

    CRITICAL: You must create exactly one of {REVIEW_ACCEPTED_FILE} or
    {REVIEW_REJECTED_FILE}."""))
    return interpolate_template(template, state)


def generate_synthesize_prompt(state: RWRState) -> str:
    """Generate the prompt for the SYNTHESIZE phase (final report generation)."""
    template = get_template('synthesize', state, lstrip_lines(
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
    2. Read {RESEARCH_PLAN_FILE} for topic hierarchy
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

    OUTPUT: Create reports/$research_name/report.md"""))
    return interpolate_template(template, state)


def generate_final_review_prompt(state: RWRState) -> str:
    """Generate the prompt for the FINAL REVIEW phase."""
    template = get_template('final_review', state, lstrip_lines(
        f"""You are conducting a final quality review of the research report.

    TOPIC:
    $original_topic

    PHASE: FINAL REVIEW
    Your goal is to ensure the report meets quality standards.

    INSTRUCTIONS:
    1. Read the synthesized report reports/$research_name/report.md
    2. Evaluate for:
       - Completeness: Does it address all topics from {RESEARCH_PLAN_FILE}?
       - Coherence: Is the writing logical and well-structured?
       - Quality: Are findings supported by evidence and properly cited?
       - Professionalism: Is the tone appropriate and free of errors?
    3. Create exactly one of:
       - {REVIEW_ACCEPTED_FILE}: report is satisfactory
       - {REVIEW_REJECTED_FILE}: quality issues requiring revision

    OUTPUT:
    - If ACCEPTED: Create {REVIEW_ACCEPTED_FILE}, research is complete
    - If REJECTED: Create {REVIEW_REJECTED_FILE} with specific issues, proceed
    to REVISE phase

    CRITICAL: You must create exactly one of {REVIEW_ACCEPTED_FILE} or
    {REVIEW_REJECTED_FILE}."""))
    return interpolate_template(template, state)


def generate_revise_prompt(state: RWRState) -> str:
    """Generate the prompt for the REVISE phase."""
    template = get_template('revise', state, lstrip_lines(
        f"""You are revising the research report to address quality issues.

    TOPIC:
    $original_topic

    PHASE: REVISE
    Your goal is to address the quality issues identified in {REVIEW_REJECTED_FILE}.

    INSTRUCTIONS:
    1. Read the critique in {REVIEW_REJECTED_FILE}
    2. Make targeted revisions to reports/$research_name/report.md
    3. Do NOT add new research or go beyond the original scope
    4. Create {REVIEW_ACCEPTED_FILE} when revisions are complete

    OUTPUT: Update reports/$research_name/report.md with revisions"""))
    return interpolate_template(template, state)


def generate_recovery_prompt(state: RWRState) -> str:
    """Generate the prompt for the RECOVERY phase."""
    template = get_template('recovery', state, lstrip_lines(
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
    4. Create a recovery plan in {RECOVERY_NOTES_FILE}

    DIAGNOSTIC STEPS:
    - Check for file system issues (permissions, disk space)
    - Verify git repository state
    - Look for configuration or environment problems
    - Check for timeout or network issues

    RECOVERY OUTPUT ({RECOVERY_NOTES_FILE}):
    - Root cause analysis
    - If it was a timeout:
        - Provide guidance for continuing where the previous phase left off
        - Suggest ways to ensure work is broken up enough to avoid another timeout
    - If it was NOT a timeout:
        - Specific steps to resolve the issue
        - Prevention measures for future iterations
        - Whether to retry the failed phase or continue with adjustments

    GOAL: Provide concise, actionable recovery guidance to get research back
    on track."""))
    return interpolate_template(template, state)


def execute_research_phase(state: RWRState) -> Result[str]:
    """Execute the RESEARCH phase."""
    if not state.lock_token or not check_project_lock(state):
        return failure(Exception('lost the lock; aborting loop'))

    print(f"Starting RESEARCH phase")
    state.current_phase = Phase.RESEARCH.value

    try:
        prompt = generate_research_prompt(state)

        result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if result.success:
            print("RESEARCH phase completed successfully")
        return result

    except Exception as e:
        error_msg = f"RESEARCH phase error: {e}"
        print(f"ERROR: {error_msg}")
        return failure(Exception(error_msg))


def execute_review_phase(state: RWRState, is_final: bool = False) -> Result[str]:
    """Execute the REVIEW phase."""
    if not state.lock_token or not check_project_lock(state):
        return failure(Exception('lost the lock; aborting loop'))

    print(f"Starting {'FINAL ' if is_final else ''}REVIEW phase")
    state.current_phase = Phase.FINAL_REVIEW.value if is_final else Phase.REVIEW.value

    try:
        if is_final:
            prompt = generate_final_review_prompt(state)
        else:
            prompt = generate_review_prompt(state)

        result = call_opencode(
            prompt, state.review_model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if result.success:
            accepted_path = get_research_path(state.research_name, REVIEW_ACCEPTED_FILE)
            rejected_path = get_research_path(state.research_name, REVIEW_REJECTED_FILE)

            if is_final:
                accepted_exists = accepted_path.exists()
                rejected_exists = rejected_path.exists()

                if accepted_exists ^ rejected_exists:
                    print("FINAL REVIEW phase completed successfully")
                    return result
                else:
                    return failure(Exception("FINAL REVIEW must create exactly "+
                        f"one of: {REVIEW_ACCEPTED_FILE}, {REVIEW_REJECTED_FILE}"))
            else:
                accepted_exists = accepted_path.exists()
                rejected_exists = rejected_path.exists()

                if accepted_exists ^ rejected_exists:
                    print("REVIEW phase completed successfully")
                    return result
                else:
                    return failure(Exception("REVIEW phase must create exactly "+
                        f"one of: {REVIEW_ACCEPTED_FILE}, {REVIEW_REJECTED_FILE}"))
        return result

    except Exception as e:
        error_msg = f"REVIEW phase error: {e}"
        print(f"ERROR: {error_msg}")
        return failure(Exception(error_msg))


def execute_synthesize_phase(state: RWRState) -> Result[str]:
    """Execute the SYNTHESIZE phase."""
    if not state.lock_token or not check_project_lock(state):
        return failure(Exception('lost the lock; aborting loop'))

    print("Starting SYNTHESIZE phase")
    state.current_phase = Phase.SYNTHESIZE.value

    try:
        prompt = generate_synthesize_prompt(state)

        result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if result.success:
            print("SYNTHESIZE phase completed successfully")
        return result

    except Exception as e:
        error_msg = f"SYNTHESIZE phase error: {e}"
        print(f"ERROR: {error_msg}")
        return failure(Exception(error_msg))


def execute_revise_phase(state: RWRState) -> Result[str]:
    """Execute the REVISE phase."""
    if not state.lock_token or not check_project_lock(state):
        return failure(Exception('lost the lock; aborting loop'))

    print("Starting REVISE phase")
    state.current_phase = Phase.REVISE.value

    try:
        prompt = generate_revise_prompt(state)

        result = call_opencode(
            prompt, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if result.success:
            print("REVISE phase completed successfully")
        return result

    except Exception as e:
        error_msg = f"REVISE phase error: {e}"
        print(f"ERROR: {error_msg}")
        return failure(Exception(error_msg))


def execute_recovery_phase(state: RWRState) -> Result[str]:
    """Execute the RECOVERY phase."""
    if not state.lock_token or not check_project_lock(state):
        return failure(Exception('lost the lock; aborting loop'))

    print(f"Starting RECOVERY phase for {state.failed_phase}")
    state.current_phase = Phase.RECOVERY.value

    try:
        sleep(RECOVERY_WAIT_SECONDS)

        prompt = try_except(generate_recovery_prompt, state)
        if not prompt.success:
            return prompt

        result = call_opencode(
            prompt.data, state.model, state.lock_token or "", timeout=state.timeout,
            mock_mode=state.mock_mode
        )

        if result.success:
            print("RECOVERY phase completed successfully")
        return result

    except Exception as e:
        error_msg = f"RECOVERY phase error: {e}"
        print(f"ERROR: {error_msg}")
        return failure(Exception(error_msg))


def retry_failed_phase(state: RWRState, failed_phase: str) -> Result[str]:
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
        return failure(Exception(f"Invalid phase for retry: {failed_phase}"))


def handle_phase_failure(
        state: RWRState, failed_phase: str, error: Exception
    ) -> Result[str]:
    """Handle phase failure with retry logic and recovery."""
    print(f"ERROR: {failed_phase} phase failed: {error}")
    if not state.lock_token or not check_project_lock(state):
        return failure(Exception('lost the lock; aborting loop'))

    state.failed_phase = failed_phase
    state.last_error = str(error)

    log_path = get_research_path(state.research_name, LOGS_DIR, "errors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{time()}: {failed_phase} failed: {error}\n")

    state.retry_count = 0
    recovery_result = None

    while state.retry_count < PHASE_RETRY_LIMIT:
        state.retry_count += 1

        if recovery_result is None or not recovery_result.success:
            print(f"Attempting RECOVERY phase for {failed_phase} (attempt "+
                f"{state.retry_count}/{PHASE_RETRY_LIMIT})...")
            recovery_result = execute_recovery_phase(state)

        if recovery_result.success:
            state.phase_recovered = True
            print(f"Recovery successful, retrying {failed_phase} (attempt "+
                f"{state.retry_count}/{PHASE_RETRY_LIMIT})...")

            retry_result = retry_failed_phase(state, failed_phase)

            state.phase_recovered = False

            if retry_result.success:
                print(f"Phase retry {failed_phase} succeeded on attempt "+
                    f"{state.retry_count}")
                return success(f"Phase {failed_phase} recovered on attempt "+
                    f"{state.retry_count}")
            else:
                print(f"Retry {state.retry_count} failed for {failed_phase}: "+
                    f"{retry_result.error}")
        else:
            print(f"Recovery failed for {failed_phase} (attempt "+
                f"{state.retry_count}/{PHASE_RETRY_LIMIT}), trying again...")
            state.last_error += f" | Recovery error: {recovery_result.error}"
            continue

    print(f"Phase {failed_phase} failed after {state.retry_count} retry attempts")
    return failure(Exception(
        f"Phase {failed_phase} failed after {state.retry_count} attempts: {error}"
    ))


def check_for_completion(state: RWRState) -> bool:
    """Check for the COMPLETED_FILE."""
    try:
        completed_file = get_research_path(state.research_name, COMPLETED_FILE)
        if completed_file.exists():
            return True
    except:
        pass
    return False


def get_completion_stats(state: RWRState) -> Result[tuple[int, int]]:
    """Get completion statistics from RESEARCH_PLAN_FILE."""
    try:
        plan_file = get_research_path(state.research_name, RESEARCH_PLAN_FILE)
        if not plan_file.exists():
            return failure(Exception("Plan file missing"))

        content = plan_file.read_text()
        total = content.count("### ")
        complete = content.count("Status: Complete")

        return success((complete, total))
    except Exception as e:
        return failure(e)


def run_final_cycle(state: RWRState) -> None:
    """Run SYNTHESIZE → REVIEW → REVISE final cycle."""
    if not state.lock_token or not check_project_lock(state):
        print("Lost the lock; aborting final cycle")
        return

    print("Running final SYNTHESIZE-REVIEW-REVISE cycle...")

    synthesize_result = execute_synthesize_phase(state)
    if not synthesize_result.success:
        print("SYNTHESIZE phase failed, exiting...")
        return

    review_result = execute_review_phase(state, is_final=True)
    if not review_result.success:
        print("FINAL REVIEW phase failed, exiting...")
        return

    rejected_path = get_research_path(state.research_name, REVIEW_REJECTED_FILE)
    if rejected_path.exists():
        revise_result = execute_revise_phase(state)
        if not revise_result.success:
            print("REVISE phase failed, exiting...")
            return

        review_result_2 = execute_review_phase(state, is_final=True)
        if not review_result_2.success:
            print("FINAL REVIEW phase after revision failed, exiting...")
            return

    print("Final cycle completed successfully!")


def generate_warning_banner(
        topics_complete: int, topics_total: int, iterations: int,
        max_iterations: int
    ) -> str:
    """Generate warning banner for incomplete research."""
    return lstrip_lines(f"""---
    ⚠️ **WARNING: ITERATION LIMIT REACHED**

    This report was generated before all research topics were completed due to
    reaching the maximum iteration limit. Some findings may be incomplete or missing.

    - Topics completed: {topics_complete} of {topics_total}
    - Iterations executed: {iterations} (limit: {max_iterations})
    - To complete this research, resume with higher --max-iterations or run
    again without the limit
    ---""")


def main_loop(state: RWRState) -> None:
    """Main RWR loop orchestrating research phases."""
    save_state_to_disk(state)
    path_dir = get_research_path(state.research_name)

    while state.iteration < state.max_iterations and not state.is_complete:
        state.iteration += 1

        result = execute_research_phase(state)
        if not result.success:
            handle_phase_failure(state, Phase.RESEARCH.value, str(result.error))

        state.phase_history.append(f"RESEARCH_{state.iteration}")
        save_state_to_disk(state)

        if (path_dir/REVIEW_ACCEPTED_FILE).exists():
            try:
                (path_dir/REVIEW_ACCEPTED_FILE).unlink()
            except Exception as e:
                print(f"WARNING: Could not delete {REVIEW_ACCEPTED_FILE}: {e}")
        if (path_dir/REVIEW_REJECTED_FILE).exists():
            try:
                (path_dir/REVIEW_REJECTED_FILE).unlink()
            except Exception as e:
                print(f"WARNING: Could not delete {REVIEW_REJECTED_FILE}: {e}")

        review_result = execute_review_phase(state)
        state.phase_history.append(f"REVIEW_{state.iteration}")
        if not review_result.success:
            handle_phase_failure(state, Phase.REVIEW.value, str(review_result.error))

        state.is_complete = check_for_completion(state)
        save_state_to_disk(state)

    complete, total = 0, 0
    result = get_completion_stats(state)
    if result.success:
        complete, total = result.data

    if state.iteration >= state.max_iterations and not state.is_complete:
        print(f"WARNING: Iteration limit ({state.max_iterations}) reached "+
            "before all topics were completed")
        print(f"Topics completed: {complete} of {total}")

    if state.is_complete or state.iteration >= state.max_iterations:
        state.phase_history.append(f"FINAL_CYCLE_{state.iteration}")
        run_final_cycle(state)

        if state.iteration >= state.max_iterations and not state.is_complete:
            report_file = Path(f"reports/{state.research_name}/report.md")
            if report_file.exists():
                banner = generate_warning_banner(
                    complete, total, state.iteration, state.max_iterations
                )
                content = report_file.read_text()
                report_file.write_text(banner + "\n" + content)
                print("Added iteration limit warning to report")

        cleanup_process_files(state)

        archive_any_process_files(state)

        release_project_lock(state)

        print("\n" + "="*60)
        print("RESEARCH COMPLETE")
        print("="*60)
        if state.iteration >= state.max_iterations and not state.is_complete:
            print("⚠️  Iteration limit reached: "+
                f"{state.iteration}/{state.max_iterations}")
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
            template_name = generate_fn.__name__.replace(
                'generate_', ''
            ).replace('_prompt', '')
            print(f"ERROR: Template '{template_name}' has invalid variable: {e}")
            return False
        except Exception as e:
            template_name = generate_fn.__name__.replace(
                'generate_', ''
            ).replace('_prompt', '')
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
        minimal_state = RWRState(research_name=args.name, lock_token=None)
        lock_file = get_research_path(args.name, LOCK_FILE)
        if lock_file.exists():
            try:
                with open(lock_file, "r") as f:
                    lock_data = json.load(f)
                minimal_state.lock_token = lock_data.get("lock_token")
            except Exception:
                pass
        reorganize_archive_files(minimal_state)
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

    # Handle resume mode
    if args.resume or args.force_resume:
        result = load_state_from_disk()
        if not result.success:
            print("ERROR: Could not load state. Run without " +
                f"{'--resume' if args.resume else '--force-resume'}.")
            return 1
        loaded_state = result.data

        lock_file = get_research_path(args.name, LOCK_FILE)
        if lock_file.exists():
            if args.force_resume:
                print("WARNING: Removing existing lock file (--force-resume)")
                lock_file.unlink()
            else:
                with open(lock_file, "r") as f:
                    lock_data = json.load(f)
                created_time = lock_data.get("created", 0)
                if time() - created_time < STALE_LOCK_TIMEOUT:
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
        if state.lock_token:
            populate_archived_hashes(state.lock_token)
        state.lock_token = None
        state.start_time = time()
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
                if response not in ('y', 'yes'):
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
            start_time=time(),
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

        populate_archived_hashes(state)

        if not check_template_generation(state):
            print("ERROR: Template validation failed")
            return 1

        print("\nGenerating initial research plan...")

        initial_plan_prompt = generate_initial_plan_prompt(state)
        result = call_opencode(
            initial_plan_prompt, state.model, state.lock_token or "",
            timeout=state.timeout, mock_mode=state.mock_mode
        )

        if not result.success:
            print(f"ERROR: Failed to generate initial plan: {result.error}")
            return 1

    save_state_to_disk(state)

    main_loop(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
