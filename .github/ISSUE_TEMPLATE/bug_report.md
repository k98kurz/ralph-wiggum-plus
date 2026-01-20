---
name: Bug report
about: Create a report to help us improve Ralph
title: 'Bug report: '
labels: bug
assignees: k98kurz

---

## Describe the bug

<!-- A clear and concise description of what the bug is. -->

## To Reproduce

Steps to reproduce the behavior:

1. Run this command:

```bash
ralph.py "Your task prompt here"
```

2. With these flags:

```bash
ralph.py --enhanced --max-iterations 10 --model opencode/big-pickle "Your task prompt here"
```

3. ...

## Expected behavior

<!-- A clear and concise description of what you expected to happen. -->

## Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

## Environment and Version Information

- OS: [e.g. Ubuntu]
- Python Version: [e.g. 3.10]
- Ralph Version: [e.g. 0.1.0]
- Build Model: [e.g. opencode/big-pickle]
- Review Model: [e.g. opencode/big-pickle]
- CLI Flags Used: [e.g. --enhanced, --mock-mode, --resume]
- Loop Mode: [basic (BUILD-PLAN) or enhanced (BUILD-REVIEW-PLAN/COMMIT)]

## Ralph State Information (Optional)

<!-- If applicable, provide information about Ralph's state files -->

<details>
<summary>State File (.ralph/state.json) (optional)</summary>

```json
{}
```

</details>

<details>
<summary>Lock File (.ralph/project.lock.json) (optional)</summary>

```json
{}
```

</details>

<details>
<summary>Iteration Count and Phase (optional)</summary>

<!-- e.g., Iteration 5, BUILD phase -->
</details>

### Additional context

<!-- Add any other context about the problem here. -->
- Was this after a crash or interruption?
- Were you resuming a previous session?
- What was in the implementation_plan.md or progress.md files?
