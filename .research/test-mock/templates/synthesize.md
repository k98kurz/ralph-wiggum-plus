You are synthesizing research findings into a comprehensive report.

    TOPIC:
    $original_topic

    PARAMETERS:
    - Breadth: $breadth
    - Depth: $depth
    - Iterations: $iteration

    PHASE: SYNTHESIZE
    Your goal is to compile all research findings into a unified report.

    INSTRUCTIONS:
    1. Read all progress/[topic_slug].md files
    2. Read research_plan.md for topic hierarchy
    3. Create reports/$research_name/report.md with:
       - Executive summary of key findings
       - Findings organized by topic hierarchy
       - Sources/references section with proper citations
       - Clear, professional writing

    REPORT FORMAT:
    # Research Report: [Topic]

    ## Executive Summary
    [Brief overview of key findings]

    ## Introduction
    [Context and research objectives]

    ## Methodology
    [Brief description of research approach]

    ## Findings

    ### [Topic 1]
    [Detailed findings with sources]

    ### [Subtopic 1.1]
    [Subtopic findings with sources]

    ## Sources
    - [Source 1]: [URL or citation]
    - [Source 2]: [URL or citation]

    ## Conclusion
    [Summary and implications]

    OUTPUT: Create reports/$research_name/report.md