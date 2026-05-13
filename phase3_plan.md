# 🤖 Phase 3 Implementation Plan: Agentic Intelligence (LangGraph)

> **Goal**: Build the multi-agent reasoning layer that makes ScholarMind *intelligent* — not just a search engine, but a system that reasons, self-corrects, and synthesizes knowledge.

---

## What Phase 3 Builds (5 Capabilities)

| # | Capability | What It Does | Needs LLM? |
|---|-----------|-------------|------------|
| **3** | Research Gap Identifier | Finds under-explored topic intersections | ❌ No (pure math) |
| **6** | Literature Review Generator | Auto-generates structured lit reviews | ✅ Yes |
| **7** | Corrective RAG + Self-Reflection | Grades retrieval, rewrites queries, catches hallucinations | ✅ Yes |
| **9** | Novelty Assessor | Scores research idea novelty vs prior art | ✅ Yes |
| **10** | Scientific Consensus Meter | Aggregates evidence across papers on a question | ✅ Yes |

---

## The LLM Cost Question

> "Wait — didn't we eliminate all LLM costs in Phase 2?"

Phase 2 was **batch processing** (50K papers). Phase 3 is **real-time user queries** — completely different usage pattern:

| Aspect | Phase 2 (Batch) | Phase 3 (Real-time) |
|--------|----------------|-------------------|
| Volume | 50,000 papers | ~30-50 queries/day |
| When | One-time pipeline run | On user request |
| Free tier fits? | ❌ No (too many calls) | ✅ Easily |

**Gemini 2.0 Flash Free Tier**:
- 15 requests per minute
- 1,500 requests per day
- Even if every user query triggers 5 LLM calls internally, that's 300 full searches/day — more than enough.

**Fallback**: If Gemini is down, we can use Ollama + Qwen2.5-7B locally.

---

## LangGraph Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│  QUERY ANALYZER  │  ← Classifies intent + extracts key terms
│  (LLM node)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CAPABILITY      │  ← Routes to the right workflow
│  ROUTER          │
│  (conditional)   │
└──┬───┬───┬───┬───┘
   │   │   │   │
   ▼   ▼   ▼   ▼
  Gap  Lit  Nov  Cons    ← Each is a sub-graph
  Ana  Rev  Ass  Meter
   │   │   │   │
   └───┴───┴───┘
         │
         ▼
┌──────────────────┐
│  CORRECTIVE RAG  │  ← Wraps ALL retrieval with self-reflection
│  LOOP            │
│  (grade → rewrite│
│   → retry × 3)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  HALLUCINATION   │  ← Final check: is the response grounded?
│  CHECKER         │
└────────┬─────────┘
         │
         ▼
     Response
```

### Shared Agent State

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Input
    query: str
    intent: str                    # "gap_analysis", "lit_review", "novelty", "consensus", "search"

    # Retrieval
    documents: list[dict]          # retrieved papers
    doc_grades: list[bool]         # per-doc relevance grades
    rewritten_queries: list[str]   # query rewrites for CRAG
    iteration_count: int           # CRAG loop counter (max 3)

    # Generation
    generation: str                # LLM output
    hallucination_check: bool      # did it pass?
    hallucination_score: float     # 0-1 grounding score

    # Sub-task specific
    sub_queries: list[str]         # for lit review decomposition
    evidence_cards: list[dict]     # for consensus meter
    novelty_scores: dict           # for novelty assessor
    gap_results: list[dict]        # for gap analyzer
```

---

## Step-by-Step: The 5 Capabilities

### Step 1: Research Gap Identifier (Capability 3)

**No LLM needed** — pure computation over BERTopic + entity co-occurrence.

**Algorithm**:
```
gap_score(topic_i, topic_j) = popularity(i) × popularity(j) / (co_occurrence(i,j) + 1)
```

- `popularity(i)` = number of papers in topic i
- `co_occurrence(i,j)` = papers that appear in BOTH topics (or that mention entities from both)
- High gap_score = popular topics that are rarely combined = **research opportunity**

**Example Output**:
```
Research Gaps Found:
1. "Reinforcement Learning" × "Medical Imaging" — gap_score: 847
   Both popular (1200+ papers each), but only 3 papers combine them.

2. "Graph Neural Networks" × "Code Generation" — gap_score: 623
   GNNs and code gen are hot, but no one uses GNNs FOR code gen.
```

**Implementation**: Pure Python/SQL query over `papers.topic_id` + `entities` table. No model.

---

### Step 2: Literature Review Generator (Capability 6)

**The most complex capability.** A LangGraph multi-step pipeline:

```
"Write a literature review on Parameter-Efficient Fine-Tuning"
    │
    ▼
┌────────────────────────────────┐
│ 1. QUERY DECOMPOSITION         │
│    → "LoRA methods"            │
│    → "Adapter architectures"   │
│    → "Prompt tuning approaches"│
│    → "Comparison studies"      │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│ 2. TARGETED RETRIEVAL (×N)     │
│    For each sub-topic:         │
│    → Hybrid search (Phase 1)   │
│    → Get top 5 papers          │
│    Total: ~20 papers           │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│ 3. INFORMATION EXTRACTION      │
│    For each paper:             │
│    → Key findings              │
│    → Methods used              │
│    → Results/metrics           │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│ 4. THEMATIC SYNTHESIS          │
│    Group findings by theme     │
│    → Identify agreements       │
│    → Identify contradictions   │
│    → Build comparative table   │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│ 5. GAP ANALYSIS                │
│    → What's missing?           │
│    → What needs more research? │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│ 6. ASSEMBLY                    │
│    → Introduction              │
│    → Thematic Sections         │
│    → Comparative Table         │
│    → Research Gaps             │
│    → References                │
└────────────────────────────────┘
```

**LLM calls**: ~6-8 per review (decomposition + extraction + synthesis + assembly).

---

### Step 3: Corrective RAG with Self-Reflection (Capability 7)

**Wraps ALL retrieval** — not a standalone feature, but a quality layer.

```
┌─────────────┐
│   RETRIEVE   │ ← Hybrid search (Phase 1 engine)
└──────┬──────┘
       ▼
┌─────────────┐     ┌──────────────┐
│  GRADE DOCS  │────→│ >60% relevant?│
│  (LLM judge) │     └───┬────┬────┘
└─────────────┘         │    │
                    YES │    │ NO
                        ▼    ▼
                  ┌────────┐ ┌──────────────┐
                  │GENERATE│ │REWRITE QUERY │
                  └───┬────┘ └──────┬───────┘
                      │            │
                      │      ┌─────┴─────┐
                      │      │iteration<3?│
                      │      └──┬────┬───┘
                      │      YES│    │NO
                      │         ▼    ▼
                      │    (loop back) (generate anyway)
                      ▼
                ┌─────────────┐
                │HALLUCINATION│
                │   CHECK     │ ← Verify every claim is grounded
                └──────┬──────┘
                       ▼
                   Response
```

**Doc Grading Prompt** (LLM):
```
Is this document relevant to answering the query?
Query: {query}
Document: {doc_title}. {doc_abstract[:300]}
Answer YES or NO with a one-line reason.
```

**Hallucination Check Prompt** (LLM):
```
For each factual claim in the response below, check if it is
supported by the provided source documents.
Response: {generation}
Sources: {documents}
Return: {"grounded": true/false, "score": 0.0-1.0, "ungrounded_claims": [...]}
```

---

### Step 4: Novelty Assessor (Capability 9)

```
User: "I want to apply LoRA to Graph Neural Networks for drug discovery"
    │
    ▼
┌──────────────────────┐
│ 1. PARSE IDEA         │ → method: LoRA, domain: GNN, application: drug discovery
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. PRIOR ART SEARCH   │ → Search for "LoRA + GNN", "LoRA + drug", "GNN + drug"
│    (3 targeted queries)│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. NOVELTY SCORING    │
│  • Method novelty:  Has LoRA been applied to GNNs? (0-10)
│  • Domain novelty:  Has this been done for drug discovery? (0-10)
│  • Combo novelty:   Has this EXACT combination been tried? (0-10)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. DIFFERENTIATION    │ → "No one has used LoRA for GNNs. Closest:
│    SUGGESTIONS        │    Adapter-GNN (2023). You could differentiate by..."
└──────────────────────┘
```

**LLM calls**: 2-3 (parse + score + suggest).

---

### Step 5: Scientific Consensus Meter (Capability 10)

```
User: "Does LoRA match full fine-tuning performance?"
    │
    ▼
┌──────────────────────┐
│ 1. RETRIEVE EVIDENCE  │ → Search for papers about LoRA vs full fine-tuning
│    (top 20 papers)     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. STANCE CLASSIFY    │ → For each paper:
│    (LLM per paper)    │    SUPPORTS / CONTRADICTS / INCONCLUSIVE
│                        │    + extract key quote
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. AGGREGATE          │
│  • 15 SUPPORT         │ → 75% agreement
│  • 3 CONTRADICT       │ → "Strong evidence"
│  • 2 INCONCLUSIVE     │
└──────────┬───────────┘
           ▼
Output: "YES — 75% agreement (Strong evidence)"
+ Evidence cards for each paper with stance + quote
```

**LLM calls**: 1 (batch all 20 papers in one prompt for stance classification).

---

## LLM Integration: Gemini 2.0 Flash

```python
# app/agent/llm.py — Shared LLM client

import google.generativeai as genai
from app.config import get_settings

def get_llm():
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def llm_call(prompt: str, json_mode: bool = False) -> str:
    model = get_llm()
    config = {}
    if json_mode:
        config["response_mime_type"] = "application/json"
    response = model.generate_content(prompt, generation_config=config)
    return response.text
```

---

## Files to Create

```
backend/app/agent/
├── __init__.py
├── llm.py                     ← Gemini client wrapper
├── state.py                   ← AgentState TypedDict
├── graph.py                   ← Main LangGraph orchestrator
├── prompts.py                 ← All LLM prompt templates
└── nodes/
    ├── __init__.py
    ├── query_analyzer.py      ← Intent classification
    ├── doc_grader.py          ← CRAG: grade document relevance
    ├── query_rewriter.py      ← CRAG: rewrite bad queries
    ├── hallucination_checker.py ← CRAG: verify grounding
    ├── review_writer.py       ← Lit review pipeline
    ├── novelty_checker.py     ← Novelty assessment
    └── consensus_builder.py   ← Consensus meter

backend/app/knowledge/
└── gap_analyzer.py            ← Research gap computation (no LLM)

backend/app/api/
├── review.py                  ← POST /api/v1/review
├── novelty.py                 ← POST /api/v1/novelty
├── consensus.py               ← POST /api/v1/consensus
└── gaps.py                    ← GET /api/v1/gaps
```

---

## New Config Values

```env
# .env additions for Phase 3
GEMINI_API_KEY=your-free-gemini-api-key
```

---

## API Endpoints

```
POST /api/v1/review          ← Generate a literature review
  Body: {"topic": "parameter efficient fine tuning", "depth": "comprehensive"}
  Response: Markdown document with sections, tables, gaps, references

POST /api/v1/novelty         ← Assess a research idea
  Body: {"idea": "Apply LoRA to GNNs for drug discovery"}
  Response: {novelty_scores, prior_art, suggestions}

POST /api/v1/consensus       ← Measure scientific consensus
  Body: {"question": "Does LoRA match full fine-tuning?"}
  Response: {verdict, agreement_pct, evidence_cards}

GET  /api/v1/gaps            ← Get research gap analysis
  Query: ?limit=20
  Response: [{topic_a, topic_b, gap_score, paper_count}]
```

---

## Execution Order

| # | Task | LLM Needed? | Depends On |
|---|------|------------|-----------|
| 1 | Write `gap_analyzer.py` | ❌ No | Phase 2 topics data |
| 2 | Write `llm.py` (Gemini client) | — | Gemini API key |
| 3 | Write `state.py` + `prompts.py` | — | None |
| 4 | Write CRAG nodes (grader, rewriter, hallucination) | ✅ | llm.py |
| 5 | Write `graph.py` (LangGraph orchestrator) | — | All nodes |
| 6 | Write `review_writer.py` | ✅ | CRAG + retrieval |
| 7 | Write `novelty_checker.py` | ✅ | CRAG + retrieval |
| 8 | Write `consensus_builder.py` | ✅ | CRAG + retrieval |
| 9 | Write API endpoints | — | All above |
| 10 | Test with real queries | ✅ | Gemini key + data |

---

## Cost Summary

| Component | Cost |
|-----------|------|
| Gemini 2.0 Flash (real-time queries) | **$0** (free tier: 1500 req/day) |
| LangGraph | **$0** (open source) |
| Gap Analyzer | **$0** (pure computation) |
| **Total** | **$0** |
