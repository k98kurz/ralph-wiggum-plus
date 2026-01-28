# Implementation Plan

## Tasks

### ANALYZE_RALPH_PATTERNS

- Status: In Review
- Description: Analyze ralph.py's specific patterns and structure that need to be mirrored
- Acceptance Criteria:
    - Document ralph.py's Result type implementation and usage patterns
    - Analyze ralph.py's constants section structure and organization
    - Document ralph.py's functional composition patterns (pipe() function usage)
    - Create specific examples of ralph.py patterns to be replicated

### ANALYZE_CURRENT_DIFFERENCES

- Status: In Review
- Description: Analyze the differences between research.py and ralph.py to understand what needs to be refactored
- Acceptance Criteria:
    - Create a detailed comparison of functional result handling between the two files
    - Identify all hard-coded filename constants in research.py
    - Document the per-research locks and directory structure features that must be preserved
    - Create a mapping of ralph.py patterns to research.py equivalents

### COMPLETE_FILENAME_CONSTANTS

- Status: Complete
- Description: Complete and organize the existing constants section in research.py
- Acceptance Criteria:
    - Review existing constants section (research.py lines 165-171)
    - Identify missing hard-coded filename strings throughout research.py
    - Add missing constants to the existing section and improve organization
    - Replace remaining hard-coded strings with constant references
    - Ensure all path-related functions use the constants

### UPDATE_RESULT_TYPE_USAGE

- Status: In Review
- Description: Update function signatures and error handling to use existing Result type system
- Acceptance Criteria:
    - Update all phase execution functions to return Result[str] instead of tuple[bool, str]
    - Replace success/error checking patterns with Result.success property checks
    - Update error handling to use Result.error instead of string messages
    - Apply pipe() function patterns from existing Result system where appropriate

### CREATE_PATH_HELPER_FUNCTION

- Status: Complete
- Description: Create a single helper function for generating per-research paths to centralize path logic
- Acceptance Criteria:
    - Create get_research_path(research_name, *path_parts) helper function
    - Replace all path construction patterns with this helper function
    - Ensure existing per-research directory structure (.research/{name}/) is preserved
    - Simplify all path generation logic to use this single helper

### UPDATE_PHASE_EXECUTION_FUNCTIONS

- Status: Pending
- Description: Update all phase execution functions to use Result types and new path helpers
- Acceptance Criteria:
    - execute_research_phase() returns Result[str] and uses path helpers
    - execute_review_phase() returns Result[str] and uses path helpers
    - execute_synthesize_phase() returns Result[str] and uses path helpers
    - execute_revise_phase() returns Result[str] and uses path helpers
    - execute_recovery_phase() returns Result[str] and uses path helpers

### UPDATE_ERROR_HANDLING_LOGIC

- Status: Pending
- Description: Update error handling throughout research.py to use Result patterns
- Acceptance Criteria:
    - handle_phase_failure() function works with Result types
    - retry_failed_phase() function works with Result types
    - main_loop() properly handles Result type returns
    - All error paths preserve existing retry/recovery logic

### UPDATE_ARCHIVING_FUNCTIONS

- Status: Pending
- Description: Update archiving functions to use new constants and path helpers
- Acceptance Criteria:
    - archive_intermediate_file() uses path helpers and constants
    - archive_any_process_files() uses path helpers and constants
    - populate_archived_hashes() uses path helpers and constants
    - reorganize_archive_files() uses path helpers and constants
    - All existing per-research archiving behavior is preserved

### UPDATE_STATE_MANAGEMENT_FUNCTIONS

- Status: Pending
- Description: Update state management functions to use new path helpers
- Acceptance Criteria:
    - save_state_to_disk() uses path helpers
    - load_state_from_disk() uses path helpers
    - validate_environment() uses path helpers
    - All existing per-research state behavior is preserved

### UPDATE_LOCKING_FUNCTIONS

- Status: Pending
- Description: Update locking functions to use new path helpers while preserving per-research locks
- Acceptance Criteria:
    - get_project_lock() uses path helpers and maintains per-research isolation
    - check_project_lock() uses path helpers and maintains per-research isolation
    - release_project_lock() uses path helpers and maintains per-research isolation
    - Per-research lock file behavior is exactly preserved

### UPDATE_TEMPLATE_FUNCTIONS

- Status: Pending
- Description: Update template functions to use new path helpers for per-research templates
- Acceptance Criteria:
    - get_template() uses path helpers for per-research template directory
    - get_global_template() remains unchanged
    - get_phase_recovery_template() remains unchanged
    - Template caching behavior is preserved



### ENSURE_MOCK_MODE_COMPATIBILITY

- Status: Pending
- Description: Ensure all test runs use mock mode as specified in requirements
- Acceptance Criteria:
    - All existing mock_mode functionality is preserved
    - call_opencode() properly handles mock_mode with Result types
    - Tests run successfully in mock mode without external dependencies
    - No changes to test files are required

### VALIDATE_REFACTORED_CODE

- Status: Pending
- Description: Validate that refactoring preserves all research.py-specific features and functionality
- Acceptance Criteria:
    - All existing test_research.py tests pass without modification
    - Per-research locks work exactly as before
    - Per-research directory structure (.research/{name}/) is preserved
    - Breadth-first search logic is unchanged
    - Research-specific phases and prompts are unchanged
    - Progress tracking and completion detection are unchanged
    - Template generation tests pass
    - Archiving system tests pass
    - Locking mechanism tests pass
    - State management tests pass
    - Mock mode tests pass
    - Integration tests pass with mock mode

## Dependencies

- **ANALYZE_RALPH_PATTERNS** depends on no other tasks
- **COMPLETE_FILENAME_CONSTANTS** depends on **ANALYZE_CURRENT_DIFFERENCES**
- **CREATE_PATH_HELPER_FUNCTION** depends on **COMPLETE_FILENAME_CONSTANTS**
- **UPDATE_RESULT_TYPE_USAGE** depends on **ANALYZE_RALPH_PATTERNS** and **ANALYZE_CURRENT_DIFFERENCES**
- **UPDATE_PHASE_EXECUTION_FUNCTIONS** depends on **UPDATE_RESULT_TYPE_USAGE** and **CREATE_PATH_HELPER_FUNCTION**
- **UPDATE_ERROR_HANDLING_LOGIC** depends on **UPDATE_PHASE_EXECUTION_FUNCTIONS**
- **UPDATE_ARCHIVING_FUNCTIONS** depends on **CREATE_PATH_HELPER_FUNCTION**
- **UPDATE_STATE_MANAGEMENT_FUNCTIONS** depends on **CREATE_PATH_HELPER_FUNCTION**
- **UPDATE_LOCKING_FUNCTIONS** depends on **CREATE_PATH_HELPER_FUNCTION**
- **UPDATE_TEMPLATE_FUNCTIONS** depends on **CREATE_PATH_HELPER_FUNCTION**
- **ENSURE_MOCK_MODE_COMPATIBILITY** depends on **UPDATE_RESULT_TYPE_USAGE**
- **VALIDATE_REFACTORED_CODE** is the final validation task depending on all implementation tasks