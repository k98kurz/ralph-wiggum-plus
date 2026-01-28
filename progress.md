# Progress Tracking

## Current Task: ANALYZE_RALPH_PATTERNS

### Learnings
- Ralph.py uses a well-structured Result type system with Ok[T] and Err classes
- All phase execution functions return Result[str] instead of tuple[bool, str]
- Ralph.py has a comprehensive constants section with clear filename organization
- The pipe() function enables functional composition patterns
- call_opencode() in ralph.py returns Result[str] while research.py returns tuple[bool, str]

### Key Ralph.py Patterns Identified:

1. **Result Type Implementation**:
   - Ok[T] dataclass with success=True and map() method
   - Err dataclass with success=False and map() method  
   - Result = Ok[T] | Err union type
   - success() and failure() helper functions
   - try_except() for exception handling

2. **Functional Composition**:
   - pipe() function for chaining operations that only continue on success
   - Each operation receives Result[T] and returns Result[T]
   - Operations stop executing on first failure

3. **Constants Organization**:
   - Configuration constants (environment variables with defaults)
   - Filename constants section with clear naming
   - All hard-coded filenames extracted to constants

4. **Phase Execution Pattern**:
   - All functions return Result[str]
   - Lock checking with failure() return on error
   - try/except blocks returning failure(Exception)
   - call_opencode() returns Result[str] directly
   - Success state printed and result returned

5. **Locking Pattern**:
   - Single project-level lock (ralph.lock.json)
   - get_project_lock() returns token or None
   - check_project_lock(token) validates lock
   - release_project_lock(token) removes lock

6. **Error Handling Pattern**:
   - Consistent use of result.success property checks
   - result.error accessed for error messages
   - result.data accessed for successful values
   - Early return patterns on failure

7. **Path Generation Pattern**:
   - Uses STATE_FILE constant (".ralph/state.json")
   - Direct Path construction with constants
   - No per-project subdirectories

### Struggles
- Need to ensure research.py's per-research directory structure is preserved
- Must maintain mock mode compatibility throughout refactoring

### Specific Examples to Replicate:

1. **Function Signature Change**:
   ```python
   # Current research.py:
   def execute_research_phase(state: RWRState) -> tuple[bool, str]:
   
   # Target ralph.py pattern:
   def execute_build_phase(state: RWLState) -> Result[str]:
   ```

2. **Error Handling Change**:
   ```python
   # Current research.py:
   if success:
       return True, result
   else:
       return False, result
   
   # Target ralph.py pattern:
   if result.success:
       print("Phase completed successfully")
   return result
   ```

3. **Lock Checking Change**:
   ```python
   # Current research.py:
   if not state.lock_token or not check_project_lock(state):
       raise Exception('lost the lock; aborting loop')
   
   # Target ralph.py pattern:
   if not state.lock_token or not check_project_lock(state.lock_token):
       return failure(Exception('lost the lock; aborting loop'))
   ```

4. **Constants Section Pattern**:
   ```python
   # Target pattern from ralph.py:
   # Filename Constants
   STATE_FILE = ".ralph/state.json"
   PROMPT_FILE = "prompt.md"
   REQUEST_REVIEW_FILE = "request.review.md"
   # ... etc
   ```

## Current Task: ANALYZE_CURRENT_DIFFERENCES

### Key Differences Identified:

1. **Return Type Patterns**:
   - research.py: `tuple[bool, str]` for all phase functions
   - ralph.py: `Result[str]` for all phase functions

2. **Constants Organization**:
   - research.py: Partial constants section (ARCHIVE_FILENAMES only)
   - ralph.py: Comprehensive filename constants section

3. **Hard-coded Strings in research.py**:
   - 29 instances of `.research/` directory paths
   - Multiple hard-coded filenames: "state.json", "research.lock.json", "progress.md", etc.
   - Hard-coded subdirectories: "/archive", "/logs", "/templates", "/progress"

4. **Per-Research Structure**:
   - research.py: `.research/{name}/` with per-research isolation
   - ralph.py: `.ralph/` single project directory

5. **Lock File Patterns**:
   - research.py: `research.lock.json` in per-research directory
   - ralph.py: `ralph.lock.json` in project root

6. **Path Generation**:
   - research.py: Scattered Path(f".research/{name}/...") throughout code
   - ralph.py: Centralized constants with simple Path construction

### Missing Hard-coded Constants:
- RESEARCH_DIR = ".research"
- LOCK_FILE = "research.lock.json"  
- STATE_FILE = "state.json"
- RESEARCH_PLAN_FILE = "research_plan.md"
- PROGRESS_FILE = "progress.md"
- REVIEW_ACCEPTED_FILE = "review.accepted.md"
- REVIEW_REJECTED_FILE = "review.rejected.md"
- RECOVERY_NOTES_FILE = "recovery.notes.md"
- COMPLETED_FILE = "completed.md"
- ARCHIVE_DIR = "archive"
- LOGS_DIR = "logs" 
- TEMPLATES_DIR = "templates"
- PROGRESS_DIR = "progress"
- REPORTS_DIR = "reports"

### Research.py Features to Preserve:
- Per-research directory structure (.research/{name}/)
- Per-research lock isolation
- Breadth-first search logic
- Research-specific phases (RESEARCH, REVIEW, SYNTHESIZE, REVISE)
- Template caching per research
- Archive system per research

### Pattern Mapping (ralph.py → research.py):

1. **Constants Section**:
   ```python
   # ralph.py:
   STATE_FILE = ".ralph/state.json"
   
   # research.py equivalent:
   RESEARCH_DIR = ".research"
   STATE_FILE = "state.json"
   # Used as: Path(f"{RESEARCH_DIR}/{research_name}/{STATE_FILE}")
   ```

2. **Lock File Pattern**:
   ```python
   # ralph.py:
   lock_file = Path("ralph.lock.json")
   
   # research.py equivalent:
   LOCK_FILE = "research.lock.json"
   # Used as: Path(f"{RESEARCH_DIR}/{research_name}/{LOCK_FILE}")
   ```

3. **Path Helper Function**:
   ```python
   # research.py needed helper:
   def get_research_path(research_name: str, *path_parts: str) -> Path:
       """Generate per-research path preserving existing structure."""
       return Path(f"{RESEARCH_DIR}/{research_name}", *path_parts)
   ```

4. **Function Return Types**:
   ```python
   # ralph.py:
   def execute_build_phase(state: RWLState) -> Result[str]:
   
   # research.py equivalent:
   def execute_research_phase(state: RWRState) -> Result[str]:
   ```

5. **Error Handling**:
   ```python
   # ralph.py:
   return failure(Exception('lost the lock; aborting loop'))
   
   # research.py equivalent:
   return failure(Exception('lost the lock; aborting loop'))
   ```

### Remaining Work
- ANALYZE_CURRENT_DIFFERENCES task complete
- COMPLETE_FILENAME_CONSTANTS task complete
- CREATE_PATH_HELPER_FUNCTION task complete
- All hard-coded .research/ paths replaced with get_research_path() helper function
- Fixed type errors with environment variable parsing (os.getenv -> int())
- All path-related functions now use centralized constants and helper function

## Current Task: COMPLETE_FILENAME_CONSTANTS

### Learnings
- Successfully replaced 4 hard-coded filename strings in main_loop() (lines 1252-1261)
- Replaced `"review.accepted.md"` with `REVIEW_ACCEPTED_FILE` constant
- Replaced `"review.rejected.md"` with `REVIEW_REJECTED_FILE` constant
- All 35 unit tests pass after the changes
- Remaining hard-coded strings (README.md, prompt.md) are either documentation files or outside per-research directory structure and don't need to be constants

### Struggles
- None

### Remaining Work
- COMPLETE_FILENAME_CONSTANTS task is complete
- All hard-coded per-research filename strings replaced with constants
- All tests pass successfully
- Next task: VALIDATE_REFACTORED_CODE (final validation)

## Current Task: UPDATE_RESULT_TYPE_USAGE

### Learnings
- Successfully updated call_opencode() to return Result[str] instead of tuple[bool, str]
- Updated all 5 phase execution functions to return Result[str]:
  - execute_research_phase()
  - execute_review_phase()
  - execute_synthesize_phase()
  - execute_revise_phase()
  - execute_recovery_phase()
- Updated error handling patterns to use result.success property checks
- Updated lock checking to return failure(Exception(...)) instead of raising
- Updated retry_failed_phase() and handle_phase_failure() to work with Result types
- Updated run_final_cycle() and main_loop() to use Result types
- All 35 unit tests pass after refactoring

### Struggles
- LSP shows false positive warnings about accessing .error attribute on Ok[str] type
- These warnings occur in handle_phase_failure() when accessing retry_result.error and recovery_result.error
- The code is correct because we only access .error when the result has failed (not result.success)
- Type checker can't infer that .error is safe in the else block

### Remaining Work
- UPDATE_RESULT_TYPE_USAGE task is essentially complete
- All phase execution functions now return Result[str]
- All error handling patterns updated to use Result.success checks
- All tests pass successfully
- Next tasks:
  - UPDATE_PHASE_EXECUTION_FUNCTIONS (dependent on UPDATE_RESULT_TYPE_USAGE)
  - UPDATE_ERROR_HANDLING_LOGIC (dependent on UPDATE_PHASE_EXECUTION_FUNCTIONS)

## Current Task: VALIDATE_REFACTORED_CODE

### Learnings
- All 35 unit tests pass without modification
- All refactoring preserves research.py-specific features:
  - Per-research locks work exactly as before
  - Per-research directory structure (.research/{name}/) is preserved
  - Breadth-first search logic is unchanged
  - Research-specific phases and prompts are unchanged
  - Progress tracking and completion detection are unchanged
- Template generation, archiving system, locking mechanism, and state management tests all pass
- Mock mode tests pass
- All hard-coded per-research filename strings replaced with constants
- Only remaining hard-coded string is prompt.md which is a project root config file, not a per-research file

### Struggles
- None

### Remaining Work
- VALIDATE_REFACTORED_CODE task is complete
- All acceptance criteria met
- All tests pass successfully
- All refactoring tasks are complete