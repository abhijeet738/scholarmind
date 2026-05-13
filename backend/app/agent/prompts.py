"""
ScholarMind — LLM Prompt Templates

All prompts used by the agentic nodes. Centralized here for
easy tuning and versioning.
"""

# ============================================================
# QUERY ANALYZER — classify user intent
# ============================================================
QUERY_ANALYZER_PROMPT = """You are a research intelligence agent. Classify the user's query intent.

Query: "{query}"

Classify into exactly ONE of these intents:
- "search" — user wants to find specific papers or topics
- "lit_review" — user wants a literature review or survey on a topic
- "novelty" — user wants to assess how novel a research idea is
- "consensus" — user wants to know scientific consensus on a question
- "gap_analysis" — user wants to find research gaps or unexplored areas

Also extract the key search terms.

Return JSON:
{{
    "intent": "one of the 5 intents above",
    "key_terms": ["term1", "term2", ...],
    "reasoning": "one line explaining why"
}}"""


# ============================================================
# DOCUMENT GRADER — CRAG relevance check
# ============================================================
DOC_GRADER_PROMPT = """You are a relevance grader for a research paper retrieval system.

User Query: "{query}"

Document Title: "{title}"
Document Abstract: "{abstract}"

Is this document relevant to answering the user's query?
Consider: Does it discuss the same topic, methods, or concepts?

Return JSON:
{{
    "relevant": true or false,
    "reason": "one line explanation"
}}"""


# ============================================================
# QUERY REWRITER — CRAG query improvement
# ============================================================
QUERY_REWRITER_PROMPT = """You are a query rewriter for an academic search engine.

The original query did not retrieve enough relevant documents.
Rewrite it to be more specific and likely to find relevant papers.

Original query: "{query}"
Previous results were about: {failed_topics}

Write a better search query. Return JSON:
{{
    "rewritten_query": "your improved query",
    "reasoning": "why this is better"
}}"""


# ============================================================
# HALLUCINATION CHECKER — verify grounding
# ============================================================
HALLUCINATION_CHECK_PROMPT = """You are a fact-checking agent for research responses.

Check if the response below is grounded in the source documents.
For each factual claim, verify it appears in at least one source.

Response to check:
---
{generation}
---

Source documents:
{sources}

Return JSON:
{{
    "grounded": true or false,
    "score": 0.0 to 1.0 (what fraction of claims are grounded),
    "ungrounded_claims": ["list of claims not found in sources"]
}}"""


# ============================================================
# LITERATURE REVIEW — decompose topic
# ============================================================
LIT_REVIEW_DECOMPOSE_PROMPT = """You are an expert research assistant writing a literature review.

Topic: "{topic}"

Decompose this topic into 4-6 sub-topics that a comprehensive literature
review should cover. Each sub-topic should be specific enough to search for.

Return JSON:
{{
    "sub_topics": [
        {{"title": "Sub-topic title", "search_query": "query to find papers"}},
        ...
    ]
}}"""


# ============================================================
# LITERATURE REVIEW — synthesize section
# ============================================================
LIT_REVIEW_SECTION_PROMPT = """You are writing one section of a literature review.

Section topic: "{section_title}"

Here are the relevant papers for this section:
{papers_text}

Write a cohesive synthesis paragraph (200-300 words) that:
1. Summarizes the key findings across these papers
2. Identifies agreements and disagreements
3. Notes methodological approaches used
4. Cites papers using [Author, Year] format

Return the section as plain text (not JSON)."""


# ============================================================
# LITERATURE REVIEW — assemble final review
# ============================================================
LIT_REVIEW_ASSEMBLE_PROMPT = """You are assembling a complete literature review.

Topic: "{topic}"

Here are the drafted sections:
{sections_text}

Assemble a complete, polished literature review in Markdown with:
1. **Introduction** — brief overview of the topic and why it matters
2. **Thematic Sections** — one per sub-topic (use the drafted sections)
3. **Comparative Summary** — a markdown table comparing key methods/results
4. **Research Gaps** — 3-5 open questions or under-explored areas
5. **References** — list all cited papers

Write in academic style. Return the full Markdown document."""


# ============================================================
# NOVELTY ASSESSOR — parse idea
# ============================================================
NOVELTY_PARSE_PROMPT = """You are a research novelty assessor.

Research idea: "{idea}"

Parse this idea into its components. Return JSON:
{{
    "method": "the core method or technique proposed",
    "domain": "the research domain or field",
    "application": "the specific application or use case",
    "search_queries": [
        "query to find papers about the method",
        "query to find papers about the domain+application",
        "query to find papers combining method+domain"
    ]
}}"""


# ============================================================
# NOVELTY ASSESSOR — score novelty
# ============================================================
NOVELTY_SCORE_PROMPT = """You are scoring the novelty of a research idea.

Research Idea: "{idea}"
- Method: {method}
- Domain: {domain}
- Application: {application}

Here is the closest prior art found:
{prior_art_text}

Score novelty on three dimensions (0 = completely exists, 10 = totally new):

Return JSON:
{{
    "method_novelty": {{
        "score": 0-10,
        "explanation": "Has this method been used before? In what context?"
    }},
    "domain_novelty": {{
        "score": 0-10,
        "explanation": "Has this been applied in this domain before?"
    }},
    "combination_novelty": {{
        "score": 0-10,
        "explanation": "Has this exact combination been tried?"
    }},
    "overall_novelty": 0-10,
    "differentiation_suggestions": [
        "suggestion 1 to make this more novel",
        "suggestion 2"
    ]
}}"""


# ============================================================
# CONSENSUS METER — classify stance
# ============================================================
CONSENSUS_STANCE_PROMPT = """You are analyzing scientific consensus on a research question.

Question: "{question}"

For each paper below, classify its stance on the question:

{papers_text}

For each paper, return:
- stance: "SUPPORTS", "CONTRADICTS", or "INCONCLUSIVE"
- key_quote: the most relevant sentence from the abstract
- confidence: 0.0 to 1.0

Return JSON:
{{
    "evidence": [
        {{
            "paper_id": "...",
            "title": "...",
            "stance": "SUPPORTS/CONTRADICTS/INCONCLUSIVE",
            "key_quote": "relevant sentence",
            "confidence": 0.0-1.0
        }},
        ...
    ]
}}"""


# ============================================================
# CONSENSUS METER — final verdict
# ============================================================
CONSENSUS_VERDICT_PROMPT = """Based on this evidence breakdown:

- SUPPORTS: {supports_count} papers
- CONTRADICTS: {contradicts_count} papers
- INCONCLUSIVE: {inconclusive_count} papers

Question: "{question}"

Provide a final verdict. Return JSON:
{{
    "verdict": "YES/NO/MIXED",
    "agreement_percentage": 0-100,
    "evidence_strength": "Strong/Moderate/Weak/Insufficient",
    "summary": "2-3 sentence summary of the consensus"
}}"""
