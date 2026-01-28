The research.py script must be refactored to mirror the ralph.py script.
Specifically, the functional result handling must be implemented in a
similar way, and hard-coded filename constants must be extracted into a
constants section near the top of the file. Do not alter the ralph.py
script or tests in any ways. Ensure that all test runs use mock mode.
Note that there are specific features of research.py which are different
from ralph.py that must be preserved, such as per-research locks and a
per-research directory structure. The paths should not be changed. A
helper function for generating per-research paths will be helpful for
centralizing and standardizing the path generation logic.
