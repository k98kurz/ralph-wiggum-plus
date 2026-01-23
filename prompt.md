The goal is to create a research tool based upon the structure of ralph.py,
called research.py. The general concept is to replace coding tasks with research
tasks, replacing the BUILD-PLAN and BUILD-REVIEW-PLAN/COMMIT cycles with a
RESEARCH-REVIEW-SYNTHESIZE cycle. The tool will instruct the AI agent in the
RESEARCH phase to do background research on the original topic/question. The
high-level tasks will be to collect general information on the topic, identify
subtopic, and collect information on those subtopics. At the end, a report will
be synthesized.

# Filename Changes

- `implementation_plan.md` -> `research_plan.md`
- `.ralph/*` -> `.research/*`
- all process files will be written in a new `reports/{name}` folder rather than in
the current working directory (see `--name` CLI parameter)

# CLI Parameters

- `--breadth X`: defaults to 3
- `--depth Y`: defaults to 3
- `--name Z`: name for the research session (required; used in file naming)
- `--update`: mode that reviews an existing report and prepares a plan to
validate and update its contents (shorter process)
- `[prompt]` -> `[topic]`: original topic to research or question to be answered

# Phases/Process

## Research Mode

1. Initial planning should be a single pass which sets up the plan incorporating
the breadth and depth parameters.
2. The RESEARCH phase has two different phases:
    - Choose a high-level research topic that is marked "Pending" or "In Progress",
    if there is one; compile notes from doing X searches on the topic; identify up to
    X subtopics to explore; write/update a `research_progress.md` file with what it
    found and information to direct research on the subtopics; mark the topic as
    "In Review"; then exit.
    - Choose a lowwer-level research subtopic that is marked "Pending" or "In Progress",
    if there is one; compile notes from doing X searches on the topic; if this subtopic
    was at depth < Y, identify X additional subtopics to explore; write/update the
    `research_progress.md` file with what it found and information to direct research
    on the sub-topics (if any); mark the subtopic as "In Review"; then exit.
3. The REVIEW phase will look at the `research_plan.md` file for anything marked
"In Review"; review the `research_progress.md` file for the topic/subtopic to review;
perform a knowledge gap analysis, identifying any weak areas that require further
research; write a `research_review.rejected.md` file with a helpful critique if there
are significant knowledge and update the topic/subtopic status back to "In Progress",
or write a `research_review.accepted.md` file if it was acceptable and mark the task
as "Complete" in the `research_plan.md` file; then exit.
