# Enhanced RWL Implementation Plan

## Overview
Single-file Python implementation of enhanced RWL loop with four-phase cycle (BUILD-REVIEW-PLAN-COMMIT) and optional recovery mechanisms.

## Important Note

While building, under no circumstance is it acceptable to test ralph.py in a way that
executes the opencode command. The only acceptable things to test are the prompt
generation, file creation/deletion/updating logic (non-agentic), and state transition
logic.

## Architecture

### Startup Logic
- If `implementation_plan.md` is missing, generate one
  - PLAN → REVIEW → PLAN → REVIEW → PLAN
  - Uses special prompt templates

### Core Loop Logic
- **Classic Mode** (`--enhanced NOT set`): BUILD → PLAN → BUILD → PLAN → ...
- **Enhanced Mode** (`--enhanced set`): BUILD → (PLAN or (REVIEW → PLAN/COMMIT)) → ... → REVIEW (final) -> BUILD -> COMMIT

### Cleanup Logic
- Delete lock file
- If the loop ended with completion rather than by hitting max iterations, delete any lingering process files:
    - request.review.md
    - review.rejected.md
    - review.passed.md
    - review.final.md
    - progress.md
    - recovery.notes.md

### Phase Flow States
```
BUILD Phase:
├── Chooses and works on a task
├── Summarizes notes (what it learned, what it struggled with, what remains to be done in the task if anything) in progress.md
├── Stages changed files with `git add`
├── Creates request.review.md (when task complete in enhanced mode)
└── Triggers REVIEW phase (in enhanced mode)

REVIEW Phase (enhanced mode only):
└── Reads request.review.md content if it exists
  ├── Creates review.passed.md OR review.rejected.md
  ├── If passed → triggers COMMIT phase
  └── If rejected → triggers PLAN phase
└── Conducts final review if `--final-review` was passed


PLAN Phase:
├── Reads review.rejected.md content if it exists
├── Reads progress.md if it exists
├── Updates implementation_plan.md
└── Returns to BUILD phase

COMMIT Phase (enhanced mode only):
├── Reads review.passed.md content
├── Stages appropriate files that were not staged in a previous step
├── Creates commit message
└── Executes git commit

RECOVERY Phase (enhanced mode only):
├── Reads .ralph state files
├── Diagnoses failure
├── Creates recovery notes
└── Returns to failed phase for retry
```

## File Structure (while ralph.py is running in a project)
```
project-root/
├── prompt.md                         # Original user prompt
├── implementation_plan.md             # Task tracking
├── request.review.md                  # Created by BUILD, deleted by REVIEW
├── review.passed.md                  # Created by REVIEW, deleted by COMMIT
├── review.rejected.md                # Created by REVIEW, deleted by PLAN
├── ralph.lock.json                     # Project lock file
└── .ralph/                          # Runtime state directory
    ├── state.json                    # Current state (backup for crash recovery)
    ├── logs/                        # Logs for debugging/tracing activities
    └── recovery.notes.md            # RECOVERY phase output
```

## Key Components

### 1. Configuration Constants
```python
DEFAULT_MODEL = "zen/big-pickle"
DEFAULT_REVIEW_MODEL = "zen/big-pickle"
OPENCODE_TIMEOUT = 600
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_REVIEW_EVERY = 0
PHASE_RETRY_LIMIT = 3
RETRY_WAIT_SECONDS = 30
RECOVERY_WAIT_SECONDS = 10
```

### 2. Locking Mechanism
- `get_project_lock()`: Creates `ralph.lock.json` with `token_hex(16)` and timestamp
- `check_project_lock()`: Reads `ralph.lock.json` on each iteration; exits if file doesn't exist or contains wrong token
- `release_project_lock()`: Removes lock file on normal exit
- Prevents multiple RWL instances in same directory
- Includes stale lock detection (locks older than 1 hour considered stale)

### 3. State Management
- In-memory tracking with `RWLState` dataclass
- File backup to `.ralph/state.json` for crash recovery
- Tracks: iteration, current_phase, failed_phase, retry_count, lock_token, start_time
- State persistence occurs after each phase completion
- Recovery state includes: phase history, error log, recovery notes

### 4. Meta-Prompt Generators (separate functions)
- `generate_build_prompt()`: Embeds original prompt + phase instructions
  - Includes iteration context, file existence checks, task selection guidance
  - Includes instructions to create a `completed.md` file with contents "<promise>COMPLETE</promise>"
    if there is nothing left to do in the `implementation_plan.md` plan file
  - Includes instructions to mark individual tasks as "Done" in the `implementation_plan.md` file
    as they are completed
  - Includes instructions to write learnings, struggles, etc to a `progress.md` file
  - Enhanced mode: instructions to create `request.review.md` when a task is complete
  - Classic mode: simpler task completion instructions
- `generate_review_prompt()`: Review criteria and output requirements
  - Reads `request.review.md` content and provides structured review framework
  - Must output either `review.passed.md` or `review.rejected.md`
  - Final review mode: different instructions for final polish cycle
- `generate_plan_prompt()`: Gap analysis and plan updates
  - Incorporates `review.rejected.md` feedback and `progress.md` learnings
  - Updates `implementation_plan.md` with prioritized tasks
- `generate_commit_prompt()`: File staging and commit logic
  - Reads `review.passed.md` and stages appropriate files
  - Generates conventional commit messages
- `generate_recovery_prompt()`: Error diagnosis and recovery steps
  - Analyzes failed phase output and system state
  - Provides specific recovery recommendations

### 5. Phase Execution Functions
- `execute_build_phase()`: Task selection, implementation, commit
  - Reads `prompt.md` and `implementation_plan.md`
  - Selects highest priority task from plan
  - Creates/updates `progress.md` with learnings and struggles
  - Stages files with `git add`
  - Enhanced mode: creates `request.review.md` when task complete
  - Executes commit and returns success/failure status
- `execute_review_phase()`: Review with file output verification
  - Checks for `request.review.md` existence, waits `RECOVERY_WAIT_SECONDS` seconds if missing
  - Reads and deletes `request.review.md`
  - Generates review with structured criteria
  - Must create exactly one of: `review.passed.md` or `review.rejected.md`
  - Retries if neither file created (no explicit error handling for this)
- `execute_plan_phase()`: Plan updates from review rejection
  - Reads `review.rejected.md` and `progress.md` if they exist
  - Updates `implementation_plan.md` with new priorities and tasks
  - Deletes `review.rejected.md` after processing
- `execute_commit_phase()`: File staging and commit logic
  - Reads `review.passed.md` content
  - Stages files not already staged (checks git status)
  - Creates conventional commit message incorporating review feedback
  - Executes git commit, optionally pushes if configured
  - Deletes `review.passed.md` after successful commit
- `execute_recovery_phase()`: Error diagnosis in enhanced mode
  - Waits `RECOVERY_WAIT_SECONDS` seconds in case there was a transient issue
  - Reads `.ralph/state.json`, `.ralph/logs/`, and failure context
  - Analyzes what went wrong and why
  - Creates recovery plan in `.ralph/recovery.notes.md`
  - Provides specific steps to resolve issue and prevent recurrence
- `execute_final_review_phase()`: Final review for polish
  - Conducts a final review, placing it in `review.final.md`
  - Quick QA pass for code style/conventions, inconsistent docblocks/docstrings, etc

### 6. CLI Interface
Arguments:
- `--max-iterations` (default: 20)
- `--skip-tests` (flag)
- `--review-every X` (default: 0)
- `--enhanced` (flag: enables four-phase cycle)
- `--final-review` (flag: final review cycle)
- `--model` (default: "zen/big-pickle")
- `--review-model` (default: "zen/big-pickle")
- `--push-commits` (flag: enables push at end of COMMIT phase)

### Data Structures
```python
@dataclass
class RWLState:
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

class Phase(Enum):
    BUILD = "BUILD"
    REVIEW = "REVIEW"
    PLAN = "PLAN"
    COMMIT = "COMMIT"
    RECOVERY = "RECOVERY"
```

### Critical Control Functions

```python
def should_trigger_review(state: RWLState) -> bool:
    """Determine if REVIEW phase should be triggered."""
    # Enhanced mode: check for request.review.md file
    if state.enhanced_mode and Path("request.review.md").exists():
        return True

    # Check review-every parameter
    if state.review_every > 0 and state.iteration > 0:
        return state.iteration % state.review_every == 0

    return False

def handle_phase_failure(state: RWLState, failed_phase: str, error: str) -> RWLState:
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

def run_final_review_cycle(state: RWLState) -> None:
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

def check_for_completion(state: RWLState) -> bool:
    """Checks for a completed.md file."""
    try:
        with open("completed.md", "r") as f:
            return "<promise>COMPLETE</promise>" in f.read()
    except:
        return False
```

### Main Loop Algorithm
```python
def main_loop(state: RWLState) -> None:
    """Main RWL loop orchestrating all phases."""
    save_state_to_disk(state)

    while state.iteration < state.max_iterations and not state.is_complete:
        # Always run BUILD phase
        success, result = execute_build_phase(state)
        if not success:
            state = handle_phase_failure(state, Phase.BUILD.value, result)
            continue

        state.iteration += 1

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

    # Check for final review condition
    if state.enhanced_mode and state.iteration < state.max_iterations and state.final_review_requested:
        run_final_review_cycle(state)

    # Save final state
    save_state_to_disk(state)
```

If the project directory already contains a prompt.md file, it will be read if
the script was invoked without a prompt argument. Otherwise, it will immediately
exit with an error stating that a prompt is required.

The tool will also expect an `implementation_plan.md` file to exist, otherwise
it will first PLAN → REVIEW → PLAN → REVIEW → PLAN to generate it.

### File Validation and Setup
- On startup: Validate `prompt.md` and `implementation_plan.md` exist
- Check for valid git repository (`.git` directory exists)
- Verify `.ralph/` directory can be created/written to
- Validate lock file doesn't exist or is stale
- Create `.ralph/logs/` directory structure if missing

### OpenCode Integration Details
- Subprocess calls: `opencode run --model <model> "<prompt>"`
- Timeout: 10 minutes per phase (configurable)
- Capture both stdout and stderr for logging
- Handle API key errors and network timeouts

## Type Annotations
Python 3.10+ syntax throughout:
- Use `tuple[bool, str] | None`
- DO NOT use `Optional[Tuple[bool, str]]`
- Modern union syntax for all type hints
- Dataclasses for structured data
- DO NOT `import from typing` or `import from types`

## Error Handling
- Retry logic for phase failures (max 3 retries; 30 second wait between retries)
- Enhanced mode: RECOVERY phase for systematic error diagnosis when a phase fails with explicit error
- Graceful degradation on repeated failures
- Comprehensive logging for debugging to `.ralph/logs/phase_<timestamp>.log`
- Signal handling for graceful shutdown (Ctrl+C)
- Cleanup of temporary files on exit
- Stale lock detection and cleanup (locks older than 1 hour)

### Logging Strategy
- Each phase writes detailed logs to `.ralph/logs/<phase>_<iteration>_<timestamp>.log`
- Logs include: phase input, AI output, success/failure status, duration
- Recovery phase logs include diagnostic information and recommendations
- Central error log in `.ralph/logs/errors.log` for critical issues
- Performance metrics tracking (phase duration, total runtime)

### Testing Constraints
During development, only these components may be tested:
- Meta-prompt generation functions (unit tests with sample inputs)
- File creation/deletion/updating logic (mock file system operations)
- State transition logic (simulated phase sequences)
- Lock mechanism functionality (token generation and validation)

Explicitly prohibited:
- Running opencode commands or any external AI tool calls
- Integration testing with actual git repositories
- End-to-end testing of complete phase cycles

## Final Review Integration
`--final-review` flag runs: REVIEW → BUILD → COMMIT cycle after main loop completion
- Provides quality gate before final delivery
- Uses similar phase mechanisms as main loop, but with the caveat that the BUILD
  phase will be instructed to address the concerns of the REVIEW output rather
  than working on an implementation task from `implementation_plan.md`
- Ensures polished final output
- Run as an additional iteration as long as the loop ended from completion rather than hitting `max_iterations`

## Self-Contained Design
- Single Python file (`ralph.py`)
- No external dependencies (stdlib only)
- Copy to global location for system-wide use
- Works in any project directory

## Implementation Requirements
- Do not run or test during development
- Focus on robust meta-prompt generation
- Ensure proper file cleanup and error handling
- Implement crash recovery state persistence
- Use modern Python type annotations throughout

## Success Criteria

The project will be assessed by a human engineer for the following:

1. Single file implementation works as standalone CLI
2. Classic and enhanced modes function correctly
3. Lock mechanism prevents concurrent instances
4. State persistence survives crashes
5. All phase transitions work as specified
6. Error recovery mechanisms are robust
7. Meta-prompts generate appropriate AI instructions

Do NOT attempt to test for these success criteria. The ONLY testing that is acceptable
will be for prompt generation functions, non-agentic file creation/deletion/updating
methods, and state transition logic.
