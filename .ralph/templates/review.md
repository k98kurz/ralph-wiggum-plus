You are reviewing the completion of a task in the development cycle.

    ORIGINAL PROMPT:
    $original_prompt

    PHASE: REVIEW
    Your goal is to evaluate if the completed task meets quality standards.

    INSTRUCTIONS:
    1. Read request.review.md to understand what was completed
    2. Review the implementation changes (check git status, diff staged files)
    3. Evaluate against project requirements and best practices
    4. Create either review.passed.md OR review.rejected.md

    REVIEW CRITERIA:
    - Task completion: Is the task fully implemented? This is the most important
    criterion. All else pales in comparison to consistency with the task description,
    requirements, and acceptance criteria.
    - Code quality: Does it follow best practices?
    - Testing: Are appropriate and relevant tests included?
    - Documentation: Does the code have sufficient type annotations and
    docblocks/docstrings?

    OUTPUT FORMAT:
    - If PASSED:
        - Create review.passed.md with approval and any minor suggestions
        - Update the status of any relevant tasks in the implementation_plan.md
        file from "In Review" to "Done"
    - If REJECTED:
        - Create review.rejected.md with specific issues and action items
        - Update the status of the rejected tasks in the implementation_plan.md
        file from "In Review" to "In Progress"

    CRITICAL: You must create exactly one of review.passed.md or review.rejected.md.