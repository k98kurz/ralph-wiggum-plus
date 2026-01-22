# Ralph Wiggum Plus for OpenCode

An open-source iterative development tool (Free and Open Source Slopware).
Development is ongoing; I am currently integrating feedback from experimentation.
This project is not yet stable or battle tested (feedback and pull requests are
welcome). Use at your own risk.

## Features

- **Agentic Fail-Forward**: Embraces failure as a diagnostic signal to drive
  subsequent development iterations.
- **Dynamic Three-Phase Loop**: Supports both basic BUILD-PLAN and enhanced
  BUILD-REVIEW-PLAN/COMMIT workflows. Opt-in with `--enhanced`; default is
  simpler BUILD-PLAN loop.
- **Session Resumption**: Crash recovery and manual resumption via persistent
  state tracking.
- **Process Archiving**: Automatic, deduplicated archiving of all intermediate
  prompts and reviews for audit and debugging.
- **Project Locking**: JSON lock files prevent simultaneous RWL processes in the
  same project. Locks include timestamps to detect and handle stale locks from
  terminated processes.
- **Phase Recovery System**: Dedicated recovery phase for diagnosing and
  overcoming transient failures or timeouts.
- **Final Review Artifact**: When `--enhanced` mode completes successfully, a
  `review.final.md` file is left in the project root containing a final
  quality review and summary of the work completed.

## Usage

Ralph is a single-file Python script. You can drop it anywhere in your `$PATH`
(e.g., `~/.local/bin/ralph.py`) and run it directly from your project root.

```text
usage: ralph.py [-h] [--version] [--max-iterations MAX_ITERATIONS]
                [--test-instructions TEST_INSTRUCTIONS]
                [--review-every REVIEW_EVERY] [--enhanced] [--final-review]
                [--model MODEL] [--review-model REVIEW_MODEL]
                [--timeout TIMEOUT] [--push-commits] [--mock-mode] [--resume]
                [--force-resume] [--reorganize-archive]
                [prompt]

Enhanced RWL - AI Code-monkey Tool

positional arguments:
  prompt                The task description for RWL to work on

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --max-iterations MAX_ITERATIONS
                        Maximum number of iterations (default: 20)
  --test-instructions TEST_INSTRUCTIONS
                        Instructions regarding testing during BUILD phase
  --review-every REVIEW_EVERY
                        Trigger REVIEW phase every N iterations (default: 0, enhanced mode only)
  --enhanced            Enable enhanced four-phase cycle (BUILD-REVIEW-PLAN-COMMIT)
  --final-review        Run final REVIEW → BUILD → COMMIT cycle after completion
  --model MODEL         AI model to use for development (default: opencode/minimax-m2.1-free)
  --review-model REVIEW_MODEL
                        AI model to use for reviews (default: opencode/big-pickle)
  --timeout TIMEOUT     Timeout in seconds for OpenCode calls (default: 1200)
  --push-commits        Push commits to remote repository after successful commit
  --mock-mode           Use mock mode for testing (no external AI calls)
  --resume              Resume from a previous session using saved state in .ralph/state.json
  --force-resume        Force resume by removing any existing lock file (bypasses staleness check)
  --reorganize-archive  Reorganize archives into per-session directories in .ralph/archive/

Examples:
  ralph.py "Build a web scraper for news articles"
  ralph.py --enhanced "Create a REST API with tests"
  ralph.py --enhanced --final-review "Complex multi-module project"
  ralph.py --review-every 3 "Regular review cycles"
  ralph.py --mock-mode "Test without external dependencies"
  ralph.py --resume
  ralph.py --force-resume

Note also that you can also use a prompt.md file, which is preferred
for complex prompts.
```

## Configuration

Ralph uses customizable prompt templates stored in `.ralph/templates/`. On first run, templates
are created with sensible defaults. You can edit these files to customize the AI's behavior
for each phase without modifying the script.

Templates use `$variable` syntax (e.g., `$original_prompt`, `$iteration`) for state
interpolation. All templates are validated on startup; invalid variables or syntax errors are
reported before the main loop begins.

Available templates:
- `initial_plan.md` - Initial implementation plan generation
- `plan_review.md` - Plan review and critique
- `revise_plan.md` - Plan revision based on feedback
- `build.md` - Build phase prompt
- `final_build.md` - Final build/polish phase
- `review.md` - Code review phase
- `final_review.md` - Final quality review
- `plan.md` - Planning phase
- `commit.md` - Commit message generation
- `recovery.md` - Recovery from phase failures
- `phase_recovery.md` - Appended to templates when retrying after recovery

## Development Loops

- **Basic Loop**: BUILD-PLAN
- **Dynamic Three-Phase Enhanced Loop**: BUILD-PLAN or BUILD-REVIEW-PLAN/COMMIT

## Basic Concept: Agentic Fail-Forward Programming

The Ralph Wiggum technique is named after the Simpsons character who keeps trying
despite perpetual failure. In agentic AI coding, this philosophy means embracing
failures as learning opportunities rather than dead ends. The approach prioritizes
an iterative approach where failures are recycled into future attempts.

**Core Loop**: An AI agent picks a task from implementation_plan.md, implements
it, writes learnings and struggles to progress.md. Then a planning phase updates
the implementation_plan.md file as necessary to align with things learned in the
build phase. The loop then continues until the build agent signals completion, or
when the max_iterations are hit.

**Enhanced Loop**: An AI agent picks a task from implementation_plan.md, implements
it, writes learnings and struggles to progress.md, then requests review whtn the
task is comple. If rejected, the feedback informs the next planning cycle. If it
passed the review phase, the changes are committed and the loop continues.
File-based state signals (request.review.md, review.passed.md, review.rejected.md)
pass context between phases, while .ralph/state.json enables crash recovery for
unattended operation. Like the basic loop, it ends when an agent signals completion.

**Engineering Benefits**: Small, iterative changes with robust feedback loops make
debugging easier and commits bisect-friendly. Deterministic context allocation
manages "context rot" effectively. Intermediate files are archived in
`.ralph/archive/` to provide a complete audit trail of the agent's thought process.
Simpler infrastructure means predictable unit economics — models like Claude
Opus 4.5 are now capable enough that minimal orchestration works reliably.

## ISC License

Copyright (c) 2026 Jonathan Voss (k98kurz) / LogixIO Corp.

Permission to use, copy, modify, and/or distribute this software
for any purpose with or without fee is hereby granted, provided
that the above copyright notice and this permission notice appear in
all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
