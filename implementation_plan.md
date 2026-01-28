# Implementation Plan

## Tasks

### ANALYZE_RALPH_PATTERNS

- Status: Completed
- Description: Analyze ralph.py's specific patterns and structure that need to be mirrored
- Acceptance Criteria:
    - ✅ Document ralph.py's Result type implementation and usage patterns
    - ✅ Analyze ralph.py's constants section structure and organization
    - ✅ Document ralph.py's functional composition patterns (pipe() function usage)
    - ✅ Create specific examples of ralph.py patterns to be replicated

### ANALYZE_CURRENT_DIFFERENCES

- Status: Completed
- Description: Analyze the differences between research.py and ralph.py to understand what needs to be refactored
- Acceptance Criteria:
    - ✅ Create a detailed comparison of functional result handling between the two files
    - ✅ Identify all hard-coded filename constants in research.py
    - ✅ Document the per-research locks and directory structure features that must be preserved
    - ✅ Create a mapping of ralph.py patterns to research.py equivalents

### COMPLETE_FILENAME_CONSTANTS

- Status: Completed
- Description: Complete and organize the existing constants section in research.py
- Acceptance Criteria:
    - ✅ Review existing constants section (research.py lines 162-195)
    - ✅ Identify missing hard-coded filename strings throughout research.py
    - ✅ Add missing constants to the existing section and improve organization
    - ✅ Replace remaining hard-coded strings with constant references (4 instances in main_loop() lines 1252-1261)
    - ✅ Ensure all path-related functions use the constants

### UPDATE_RESULT_TYPE_USAGE

- Status: Completed
- Description: Update function signatures and error handling to use existing Result type system
- Acceptance Criteria:
    - ✅ Update all phase execution functions to return Result[str] instead of tuple[bool, str]
    - ✅ Replace success/error checking patterns with Result.success property checks
    - ✅ Update error handling to use Result.error instead of string messages
    - ✅ Apply pipe() function patterns from existing Result system where appropriate

### CREATE_PATH_HELPER_FUNCTION

- Status: Completed
- Description: Create a single helper function for generating per-research paths to centralize path logic
- Acceptance Criteria:
    - ✅ Create get_research_path(research_name, *path_parts) helper function
    - ✅ Replace all path construction patterns with this helper function
    - ✅ Ensure existing per-research directory structure (.research/{name}/) is preserved
    - ✅ Simplify all path generation logic to use this single helper

### VALIDATE_REFACTORED_CODE

- Status: Completed
- Description: Validate that refactoring preserves all research.py-specific features and functionality
- Acceptance Criteria:
    - ✅ All existing test_research.py tests pass without modification
    - ✅ Per-research locks work exactly as before
    - ✅ Per-research directory structure (.research/{name}/) is preserved
    - ✅ Breadth-first search logic is unchanged
    - ✅ Research-specific phases and prompts are unchanged
    - ✅ Progress tracking and completion detection are unchanged
    - ✅ Template generation tests pass
    - ✅ Archiving system tests pass
    - ✅ Locking mechanism tests pass
    - ✅ State management tests pass
    - ✅ Mock mode tests pass
    - ✅ Integration tests pass with mock mode
    - ✅ No hard-coded filename strings remain in codebase (remaining prompt.md is project root config file, not per-research file)

## Dependencies

- **ANALYZE_RALPH_PATTERNS** - Completed, no dependencies
- **ANALYZE_CURRENT_DIFFERENCES** - Completed, no dependencies
- **COMPLETE_FILENAME_CONSTANTS** - Completed
- **UPDATE_RESULT_TYPE_USAGE** - Completed
- **CREATE_PATH_HELPER_FUNCTION** - Completed
- **VALIDATE_REFACTORED_CODE** - Completed

## Blockers

None
