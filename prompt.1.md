The goal is to create a research tool based upon the structure of ralph.py,
called research.py. The general concept is to replace coding tasks with research
tasks, replacing the BUILD-PLAN and BUILD-REVIEW-PLAN/COMMIT cycles with a
RESEARCH-REVIEW-SYNTHESIZE cycle. The tool will instruct the AI agent in the
RESEARCH phase to do background research on the original topic/question, performing
a breadth-first search traversal with configurable breadth and depth parameters.
At the end, a report will be synthesized through an automatic final cycle.

# Filename Changes

- `implementation_plan.md` -> `research_plan.md`
- `.ralph/*` -> `.research/{name}/*`
- `ralph.lock.json` -> `research.lock.json`
- `progress.md` preserved (with per-topic `progress/[topic_slug].md` subdirectory)
- All process files will be written in a new `.research/{name}` folder rather than in
the current working directory (see `--name` CLI parameter)
- Final reports:
    - `reports/{name}/` directory for reports on researched topics
    - `report.md` (research mode)

# CLI Parameters

## Customization Parameters
- `--breadth X`: defaults to 3
- `--depth Y`: defaults to 3
- `--name Z`: name for the research session (used in directory structure `.research/{name}/` and `reports/{name}/`)
- `--max-iterations N`: maximum number of RESEARCH-REVIEW cycles (default: calculated as X^(Y+1)+5 where X=breadth, Y=depth; minimum value is enforced)
- `--reorganize-archive`: reorganize archives in `.research/{name}/archive/` -> `.research/{name}/archive/{token}/`

## Model & Execution Parameters (from ralph.py)
- `--model MODEL`: AI model to use for research (default: opencode/minimax-m2.1-free)
- `--review-model REVIEW_MODEL`: AI model to use for reviews (default: opencode/big-pickle)
- `--timeout TIMEOUT`: Timeout in seconds for OpenCode calls (default: 1200)
- `--resume`: Resume from a previous session using saved state in `.research/{name}/state.json`
- `--force-resume`: Force resume by removing any existing lock file (bypasses staleness check)
- `--mock-mode`: Use mock mode for testing (no external AI calls)

## Removed Parameters (not applicable to research)
- `--review-every`: every topic is reviewed anyway
- `--enhanced`: always uses full RESEARCH-REVIEW cycle
- `--final-review`: automatic when complete
- `--push-commits`: not applicable to research

# File Organization

## Directory Structure
```
.research/{name}/
├── research.lock.json           # Session lock file
├── state.json                   # Persistent state for crash recovery
├── research_plan.md             # Research topics tree with status tracking
├── progress.md                  # Overall research progress notes
├── progress/[topic_slug].md     # Per-topic/subtopic detailed notes
├── review.accepted.md  # Approval markers
├── review.rejected.md  # Critique with gaps to address
├── complete.md         # Marker file when research is complete
└── archive/                     # Archived intermediate files

reports/{name}/
└── report.md                 # Synthesized final report
```

## State Persistence
- `state.json` contains serialized RWRState:
```python
@dataclass
class RWRState:
    """Represents the current state of the RWR process."""
    original_topic: str = field(default="")
    breadth: int = field(default=3)
    depth: int = field(default=3)
    iteration: int = field(default=0)
    current_phase: str = field(default=Phase.RESEARCH.value)
    failed_phase: str | None = field(default=None)
    retry_count: int = field(default=0)
    max_iterations: int = field(default=DEFAULT_MAX_ITERATIONS)
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
```
- Enables crash recovery and session resumption
- Similar to ralph.py's state management

# File Format Specifications

## research_plan.md Format
```markdown
# Research Plan

## Metadata
- Topic: [original research topic or question]
- Breadth: X
- Depth: Y

## Topics

### [Topic Name] (Depth: 0)
- Status: Pending | In Progress | In Review | Complete
- Description: Brief description of the topic
- Acceptance Criteria:
  - Criterion 1
  - Criterion 2
  - Criterion 3

### [Subtopic Name] (Depth: 1)
- Status: Pending | In Progress | In Review | Complete
- Parent: [Parent Topic Name]
- Description: Brief description of the subtopic
- Acceptance Criteria:
  - Criterion 1
  - Criterion 2

## Completion Status
- Total Topics: N
- Complete: M
- In Progress: P
- Pending: Q
```

## progress.md Format (Overall)
```markdown
# Research Progress

## Summary
- Iteration: N
- Topics Completed: M of P

## Recent Activity
- [Most recent findings or reviews]
```

## progress/[topic_slug].md Format (Per-Topic)
```markdown
# Progress: [Topic Name]

## Findings
- Key finding 1 with sources
- Key finding 2 with sources

## Sources
- Source 1: [URL or citation]
- Source 2: [URL or citation]

## Knowledge Gaps
- Gap 1: [description]
- Gap 2: [description]

## Notes for Subtopics (if applicable)
- Subtopic 1: [directions for research]
- Subtopic 2: [directions for research]
```

# Phases/Process

## Overview

The tool operates in one primary mode:
- **Research Mode**: Initial research from scratch using RESEARCH-REVIEW cycles

Both modes conclude with an automatic final cycle: SYNTHESIZE-REVIEW-REVISE

## Iteration Limit Enforcement

The tool calculates a minimum safe iteration limit: `X^(Y+1) + 5`, where X=breadth and Y=depth.
- If `--max-iterations` is not provided, this calculated minimum is used
- If `--max-iterations` is provided but less than the minimum, the minimum is enforced
- If the iteration limit is reached before all topics are complete:
  - A warning is prepended to the final report file
  - The final SYNTHESIZE-REVIEW-REVISE cycle still executes
  - The warning indicates that research may be incomplete

---

## Research Mode

### Initial Planning (One-time Setup)

Generate the initial `research_plan.md` by:
1. Calculate and enforce minimum `max_iterations` as `X^(Y+1) + 5`
    - Warn the user if the calculated `max_iterations` is greater than 20
    - User must then confirm before the loop begins
2. Analyzing the topic to identify root research question(s)
3. Creating depth 0 topics for each major question (up to breadth X)
4. Setting up `.research/{name}/` and `reports/{name}/` directory structures
5. Writing `research_plan.md` in the specified format

### Main Loop: RESEARCH-REVIEW Cycle

The tool cycles through RESEARCH → REVIEW until all topics are marked "Complete" or max_iterations is reached.

#### RESEARCH Phase

Execute in this order:

1. **Check for high-level topics** (depth 0) marked "Pending" or "In Progress" (AI agent)
   - If found:
     - Compile notes from X searches on the topic using web and file search tools
     - Identify up to X subtopics to explore (depth + 1)
     - Write/update `progress/[topic_slug].md` with detailed findings
     - Update overall `progress.md` with summary
     - Update `research_plan.md` with any new subtopics (if depth < Y)
     - Mark topic status to "In Review" in `research_plan.md`
     - Increment iteration counter
     - Halt

2. **Check for lower-level topics** (depth > 0) marked "Pending" or "In Progress" (AI agent)
   - Prioritize tasks with lower depth
   - If found:
     - Compile notes from X searches on the topic using web and file search tools
     - If depth < Y: identify up to X additional subtopics (depth + 1) to explore
     - Write/update `progress/[topic_slug].md` with detailed findings
     - Update overall `progress.md` with summary
     - Update `research_plan.md` with any new subtopics (if depth < Y)
     - Mark subtopic status to "In Review" in `research_plan.md`
     - Increment iteration counter
     - Halt

3. **If all tasks are marked "Complete"** (AI agent)
    - Create `.research/{name}/completed.md` file with just the content <promise>COMPLETE</promise>
    - Halt

4. **Check for completion or iteration limit** (script)
From the current ralph.py script:
```python
def main_loop(state: RWRState):
    ...
    while state.iteration < state.max_iterations and not state.is_complete:
        state.iteration += 1
        ...
        # Check for completion promise
        state.is_complete = check_for_completion(state)
        ...
```

#### REVIEW Phase

Execute when `research_plan.md` contains items marked "In Review":

1. Look at `research_plan.md` for any item marked "In Review"
2. Review the corresponding `progress/[topic_slug].md` file for that topic/subtopic
3. Perform knowledge gap analysis identifying weak areas
4. Create exactly one of:
   - `review.rejected.md`: specific knowledge gaps and action items required, then update topic/subtopic status back to "In Progress" in `research_plan.md`
   - `review.accepted.md`: confirmation that findings are acceptable, then mark the topic/subtopic as "Complete" in `research_plan.md`
5. Halt

---

## Final Cycle: SYNTHESIZE-REVIEW-REVISE (Automatic)

This cycle runs automatically when:
- `completed.md` file is created, OR
- max_iterations has been reached

### SYNTHESIZE Phase

1. Compile all `progress/[topic_slug].md` files into a unified report
2. Create the appropriate output file: `reports/{name}/report.md`
3. Format requirements:
   - Executive summary of key findings
   - Findings organized by topic hierarchy (mirroring `research_plan.md` structure)
   - Sources/references section with proper citations
   - Clear, professional writing suitable for the target audience
4. Halt

### REVIEW Phase (Final)

1. Read the synthesized report `report.md`
2. Evaluate for:
   - Completeness: Does it address all topics from the research plan?
   - Coherence: Is the writing logical and well-structured?
   - Quality: Are findings supported by evidence and properly cited?
   - Professionalism: Is the tone appropriate and free of errors?
3. Create exactly one of:
   - `review.rejected.md`: specific quality issues requiring revision, proceed to REVISE phase
   - `review.accepted.md`: confirmation the report is satisfactory, mark research complete
4. Halt

### REVISE Phase (Conditional)

Only runs if the final REVIEW phase creates `review.rejected.md`:

1. Read the critique in `review.rejected.md`
2. Make targeted revisions to the report addressing the identified issues
3. Do NOT add new research or go back to RESEARCH phase
4. Create `review.accepted.md` when revisions are complete
5. Halt

### Automatic Archiving of Intermediate Files

Similarly to ralph.py, research.py will archive intermediate files. However, since the
BFS system includes subtopics and places progress files for each subtopic in its own
subdirectory, it will have to be adapted as follows:

1. Add a check that goes through the subdirectories of `.research/{name}/progress/`
to identify subtopics. (Add to `archive_any_process_files`.)
2. Attempt to archive the `progress.md` files in those subdirectories by prepending
the subtopic name to the filename, e.g. `{subtopic}.progress.md`, then prepend the
timestamp and token as normal. (Add optional `prepend: str` argument to
`archive_intermediate_file`.)

### Completion

When the final REVIEW phase creates `review.accepted.md`:

1. If iteration limit reached: script will prepend warning banner at top of report:
```markdown
---
⚠️ **WARNING: ITERATION LIMIT REACHED**

This report was generated before all research topics were completed due to reaching the maximum iteration limit. Some findings may be incomplete or missing.

- Topics completed: M of N
- Iterations executed: K (limit: max_iterations)
- To complete this research, resume with higher --max-iterations or run again without the limit
---
```
2. Archive intermediate files to `.research/{name}/archive/`
3. Clean up intermediate files in `.research/{name}/`, including
`.research/{name}/progress/*`
4. Release lock file
5. Display completion summary (including warning if iteration limit was reached)

# Completion Conditions

The research tool will complete successfully when:

1. Either:
   - All topics in `research_plan.md` are marked "Complete" (i.e. the AI agent
    creates the `completed.md` file), OR
   - max_iterations has been reached (with warning in report)
2. The final SYNTHESIZE-REVIEW-REVISE cycle has been executed

The RESEARCH-REVIEW loop will signal completion by creating
`<promise>COMPLETE</promise>` in `.research/{name}/completed.md`, analogous to
ralph.py's completion mechanism.

If max_iterations was reached, the report will contain a warning banner indicating
incomplete research.

# Template System

The prompt template system used in ralph.py is preserved. Only the template names,
prompt generator functions, and default templates are changed.

The tool uses customizable prompt templates stored in `.research/templates/`.
On first run, templates are created with sensible defaults.

Templates use `$variable` syntax (e.g., `$original_topic`, `$breadth`, `$depth`,
`$iteration`, `$max_iterations`, `$model`, `$review_model`, `$timeout`) for state interpolation.
All templates are validated on startup; invalid variables or syntax errors are
reported before the main loop begins.

Available templates:
- `initial_plan.md` - Initial research plan generation
- `research.md` - Research phase prompt
- `review.md` - Review phase prompt
- `synthesize.md` - Synthesis phase prompt (final report generation)
- `final_review.md` - Final quality review prompt
- `revise.md` - Report revision prompt
- `recovery.md` - Recovery from phase failures
- `phase_recovery.md` - Appended to templates when retrying after recovery

This aligns with ralph.py's template system, allowing for similar customization capabilities.

# Error Handling & Recovery

The tool preserves ralph.py's phase recovery and retry system:

## Phase Failure Recovery
- If any phase fails (timeout, error, etc.), the tool enters RECOVERY phase
- Maximum retry attempts: 3 per phase
- Recovery notes written to `.research/{name}/recovery.notes.md`
- After recovery, the failed phase is retried with updated context
- Timeout is configurable via `--timeout` parameter

## Locking Mechanism
- `research.lock.json` prevents concurrent research sessions for the same topic
- Lock includes timestamp, lock token, and process ID
- Stale locks (> 60 minutes) are automatically removed
- Lock is updated periodically during active sessions
- `--force-resume` bypasses staleness check and removes lock

## Crash Recovery
- State persisted to `state.json` after each phase
- Sessions can be resumed using `--resume` flag
- `--force-resume` removes stale lock and allows resumption
- Full phase history tracked in state for debugging

## Testing Mode
- `--mock-mode` enables testing without external AI calls
- Useful for development and debugging
- Returns mock responses instead of calling OpenCode

## Iteration Limit Handling
- If max_iterations is reached, research stops but synthesis still occurs
- Warning banner prepended to final report
- User can resume with higher iteration limit if needed

# Additional Notes

## Iteration Limit Calculation

The minimum safe iteration limit is calculated as: `X^(Y+1) + 5`

**Rationale**:
- A complete breadth-first search with depth Y and breadth X requires approximately X^(Y+1) topics
- Each topic requires at least one RESEARCH-REVIEW cycle (one iteration)
- The +5 buffer accounts for edge cases, rejected work requiring re-research, and the final SYNTHESIZE-REVIEW-REVISE cycle

**Examples**:
- Breadth=3, Depth=3: minimum = 3^(3+1) + 5 = 81 + 5 = 86 iterations
- Breadth=2, Depth=2: minimum = 2^(2+1) + 5 = 8 + 5 = 13 iterations
- Breadth=4, Depth=2: minimum = 4^(2+1) + 5 = 64 + 5 = 69 iterations

## Breadth and Depth Enforcement
- If fewer than X subtopics can be found for a topic, the agent should not force additional subtopics
- When depth Y is reached, no new subtopics should be created (leaf nodes only)
- The agent should focus on deeper research of existing topics rather than creating spurious subtopics

## Topic Selection Priority
The RESEARCH phase follows this priority order:
1. High-level topics (depth 0) marked "Pending" or "In Progress"
2. Lower-level topics (depth > 0) marked "Pending" or "In Progress"
3. Check iteration limit: if reached, proceed to synthesis
4. If no topics found and iteration limit not reached, proceed to synthesis

## File Naming Conventions
- Topic slugs for progress files: lowercase, hyphen-separated, no special characters
- Example: "Machine Learning Fundamentals" → `progress/machine-learning-fundamentals.md`

## Concurrent Research
The `.research/{name}/` structure allows multiple concurrent research sessions on different topics, but prevents concurrent modification of the same research session via the lock file.

## Model Configuration
- `--model`: Configures the AI model used for research phases (default: opencode/minimax-m2.1-free)
- `--review-model`: Configures the AI model used for review phases (default: opencode/big-pickle)
- Use higher-quality models for reviews for better quality control

# Implementation Notes for Developers

## Key Differences from ralph.py

| Aspect              | ralph.py                               | research.py                                         |
|---------------------|----------------------------------------|-----------------------------------------------------|
| Primary cycle       | BUILD-PLAN or BUILD-REVIEW-PLAN-COMMIT | RESEARCH-REVIEW + SYNTHESIZE-REVIEW-REVISE          |
| Task types          | Coding tasks                           | Research topics/subtopics                           |
| Output files        | Code changes, commits                  | Research reports                                    |
| Traversal strategy  | Priority-based                         | Breadth-first search with depth limit               |
| Directory structure | `.ralph/`                              | `.research/{name}/` + `reports/{name}/`             |
| Lock file           | `ralph.lock.json`                      | `research.lock.json`                                |
| Progress file       | progress.md                            | progress.md + progress/[topic_slug].md              |
| Plan file           | implementation_plan.md                 | research_plan.md                                    |
| Iteration limit     | Configurable, default 20               | Calculated minimum X^(Y+1)+5, configurable          |
| Final cycle         | Optional (requires `--final-review`)   | Automatic when all topics complete or limit reached |
| Parallelism         | Sequential                             | Sequential (avoid race conditions)                  |

## Preserved Features from ralph.py
- State management (`state.json`)
- Locking mechanism
- Phase recovery and retry system
- Template system with variable interpolation
- Archival of intermediate files
- `--reorganize-archive` functionality
- `--model` and `--review-model` parameters
- `--timeout` parameter
- `--resume` and `--force-resume` parameters
- `--mock-mode` parameter

## Simplified Feature Set (removed from ralph.py)
- `--review-every` (every topic is reviewed anyway)
- `--enhanced` mode (always uses RESEARCH-REVIEW cycle)
- `--final-review` flag (automatic SYNTHESIZE-REVIEW-REVISE when complete)
- `--push-commits` (not applicable to research)

## Implementation Priority
1. File structure and locking mechanism (.research/{name}/research.lock.json, .research/{name}/)
2. State management adapted for research tasks (including iteration tracking)
3. Minimum max_iteration calculation and enforcement (X^(Y+1)+5)
4. Template system initialization
5. RESEARCH-REVIEW cycle implementation
6. SYNTHESIZE-REVIEW-REVISE final cycle
7. Completion detection and reporting
8. Warning banner generation when iteration limit reached
9. CLI parameter handling (model, review_model, timeout, resume, force-resume, mock-mode, max-iterations, name)
