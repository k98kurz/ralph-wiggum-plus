# Ralph Wiggum Plus for OpenCode

A Free and Open Source Slopware project. Development is ongoing. The first test
is about to be run. I will put some actual effort into this readme after it is
working.

## Development Loops

- **Basic Loop**: BUILD-PLAN
- **Dynamic Three-Phase Enhanced Loop**: BUILD-PLAN or BUILD-REVIEW-PLAN/COMMIT

## Basic Concept: Agentic Fail-Forward Programming

The Ralph Wiggum technique is named after the Simpsons character who keeps trying
despite perpetual failure. In agentic AI coding, this philosophy means embracing
failures as learning opportunities rather than dead ends. The approach prioritizes
simplicity over complex orchestration—using a deterministic loop that mimics how
real engineers work.

**Core Mechanics**: An AI agent picks a task from implementation_plan.md, implements
it, writes learnings and struggles to progress.md, then requests review. If rejected,
the feedback informs the next planning cycle. If passed, changes are committed and
the loop continues. File-based state signals (request.review.md, review.passed.md,
review.rejected.md) pass context between phases, while .ralph/state.json enables
crash recovery for unattended operation.

**Engineering Benefits**: Small, iterative changes with robust feedback loops make
debugging easier and commits bisect-friendly. Deterministic context allocation
manages "context rot" effectively. Simpler infrastructure means predictable unit
economics — models like Claude Opus 4.5 are now capable enough that minimal
orchestration works reliably.

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
