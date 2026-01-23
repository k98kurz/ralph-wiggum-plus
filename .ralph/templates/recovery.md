You are diagnosing and recovering from a phase failure.

    ORIGINAL PROMPT:
    $original_prompt

    PHASE: RECOVERY
    Failed Phase: $failed_phase
    Error: $last_error

    CONTEXT:
    - Current iteration: $iteration
    - Max iterations: $max_iterations
    - Retry count: $retry_count

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
        - Suggest ways to ensure task work is broken up enough to avoid another timeout
    - If it was NOT a timeout:
        - Specific steps to resolve the issue
        - Prevention measures for future iterations
        - Whether to retry the failed phase or continue with adjustments

    GOAL: Provide concise, actionable recovery guidance to get development back on track.