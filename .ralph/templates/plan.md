You are planning development tasks or revising an existing plan.

    ORIGINAL PROMPT:
    $original_prompt

    PHASE: PLAN
    Your goal is to update the implementation plan based on current progress and feedback.

    CONTEXT:
    - Current iteration: $iteration
    - Max iterations: $max_iterations

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

    OUTPUT: Update implementation_plan.md with the revised plan