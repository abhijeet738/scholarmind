"""
ScholarMind — LangGraph Agent Orchestrator

The main graph that routes user queries through the correct
capability pipeline with Corrective RAG wrapping all retrieval.

Architecture:
    Query → Analyzer → Router → [Capability Node] → Hallucination Check → Response
                                       ↑
                              CRAG Loop (grade → rewrite → retry)
"""

# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes.query_analyzer import query_analyzer
from app.agent.nodes.doc_grader import doc_grader
from app.agent.nodes.query_rewriter import query_rewriter
from app.agent.nodes.hallucination_checker import hallucination_checker
from app.agent.nodes.review_writer import review_writer
from app.agent.nodes.novelty_checker import novelty_checker
from app.agent.nodes.consensus_builder import consensus_builder
from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank

# ============================================================
# Helper nodes
# ============================================================

def retrieve_node(state: AgentState) -> dict:
    """Retrieve documents using the Phase 1 hybrid search engine."""
    query = state["query"]
    try:
        dense_results = dense_search(query, top_k=50)
        sparse_results = sparse_search(query, top_k=50)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=20)
        reranked = rerank(query, fused, top_k=10)
        return {"documents": reranked}
    except Exception as e:
        return {"documents": [], "error": f"Retrieval failed: {str(e)}"}
       

def generate_node(state: AgentState) -> dict:
    """Generate a response from retrieved documents (for simple search)."""
    from app.agent.llm import llm_call

    documents = state.get("documents", [])
    query = state["query"]

    if not documents:
        return {"generation": "No relevant papers found for your query."}

    # Build context from retrieved papers
    context = "\n\n".join(
        f"**{doc.get('title', 'Untitled')}** ({doc.get('year', 'N/A')})\n"
        f"{doc.get('abstract', '')[:400]}"
        for doc in documents[:10]
    )

    prompt = (
        f"Based on these research papers, provide a comprehensive answer to the query.\n\n"
        f"Query: {query}\n\n"
        f"Papers:\n{context}\n\n"
        f"Provide a clear, well-structured answer citing the relevant papers."
    )

    try:
        response = llm_call(prompt, temperature=0.3)
        return {"generation": response}
    except Exception as e:
        # Fallback: just list the papers
        paper_list = "\n".join(
            f"- {doc.get('title', 'Untitled')} ({doc.get('year', '')})"
            for doc in documents[:10]
        )
        return {"generation": f"Found {len(documents)} relevant papers:\n\n{paper_list}"}


# ============================================================
# Routing logic
# ============================================================

def route_by_intent(state: AgentState) -> str:
    """Route to the correct capability based on classified intent."""
    intent = state.get("intent", "search")

    routing = {
        "lit_review": "review_writer",
        "novelty": "novelty_checker",
        "consensus": "consensus_builder",
        "gap_analysis": "gap_analysis",
        "search": "retrieve",
    }
    
    return routing.get(intent, "retrieve")


def should_rewrite(state: AgentState) -> str:
    """
    CRAG decision: should we rewrite the query?

    If <60% of documents are relevant AND we haven't exceeded
    max iterations, rewrite and retry. Otherwise, generate.
    """
    doc_grades = state.get("doc_grades", [])
    iteration_count = state.get("iteration_count", 0)

    if not doc_grades:
        return "generate"

    relevant_ratio = sum(doc_grades) / len(doc_grades)

    if relevant_ratio < 0.6 and iteration_count < 3:
        return "rewrite"
    else:
        return "generate"


def gap_analysis_node(state: AgentState) -> dict:
    """Run the research gap analyzer (no LLM needed)."""
    from app.knowledge.gap_analyzer import compute_research_gaps

    gaps = compute_research_gaps(limit=20)

    if not gaps:
        return {
            "generation": "No research gaps found. Make sure the database has topic assignments.",
            "gap_results": [],
        }

    # Format as readable response
    generation = "# Research Gap Analysis\n\n"
    generation += "These popular topic pairs are rarely combined — potential research opportunities:\n\n"

    for i, gap in enumerate(gaps[:15], 1):
        generation += (
            f"### {i}. {gap['topic_a']} × {gap['topic_b']}\n"
            f"- **Gap Score:** {gap['gap_score']}\n"
            f"- Topic A: {gap['topic_a_papers']} papers | "
            f"Topic B: {gap['topic_b_papers']} papers | "
            f"Combined: {gap['co_occurrence']} papers\n\n"
        )

    return {
        "generation": generation,
        "gap_results": gaps,
    }


# ============================================================
# Build the LangGraph
# ============================================================

def build_agent_graph() -> StateGraph:
    """
    Build the ScholarMind LangGraph agent.

    Flow:
    1. query_analyzer: classify intent
    2. Router: branch to capability
    3. Capability node: execute
    4. For search: CRAG loop (grade → rewrite → retry)
    5. Hallucination check
    6. END
    """
    workflow = StateGraph(AgentState)

    # ── Add nodes ──
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("doc_grader", doc_grader)
    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_checker", hallucination_checker)
    workflow.add_node("review_writer", review_writer)
    workflow.add_node("novelty_checker", novelty_checker)
    workflow.add_node("consensus_builder", consensus_builder)
    workflow.add_node("gap_analysis", gap_analysis_node)

    # ── Entry point ──
    workflow.set_entry_point("query_analyzer")

    # ── Router: intent → capability ──
    workflow.add_conditional_edges(
        "query_analyzer",
        route_by_intent,
        {
            "retrieve": "retrieve",
            "review_writer": "review_writer",
            "novelty_checker": "novelty_checker",
            "consensus_builder": "consensus_builder",
            "gap_analysis": "gap_analysis",
        },
    )

    # ── Search path: retrieve → grade → [rewrite|generate] ──
    workflow.add_edge("retrieve", "doc_grader")
    workflow.add_conditional_edges(
        "doc_grader",
        should_rewrite,
        {
            "rewrite": "query_rewriter",
            "generate": "generate",
        },
    )
    workflow.add_edge("query_rewriter", "retrieve")  # CRAG loop back
    workflow.add_edge("generate", "hallucination_checker")

    # ── Capability paths → hallucination check ──
    workflow.add_edge("review_writer", "hallucination_checker")
    workflow.add_edge("novelty_checker", "hallucination_checker")
    workflow.add_edge("consensus_builder", "hallucination_checker")

    # ── Gap analysis → END (no LLM, no hallucination risk) ──
    workflow.add_edge("gap_analysis", END)

    # ── Hallucination check → END ──
    workflow.add_edge("hallucination_checker", END)

    return workflow


def get_agent():
    """Get a compiled agent graph ready for invocation."""
    workflow = build_agent_graph()
    return workflow.compile()


async def run_agent(query: str) -> dict:
    """
    Run the full agent pipeline on a query.

    Returns the final state with generation, documents, and metadata.
    """
    agent = get_agent()
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "documents": [],
        "doc_grades": [],
        "rewritten_queries": [],
        "iteration_count": 0,
        "generation": "",
        "hallucination_check": False,
        "hallucination_score": 0.0,
        "error": None,
    }

    result = agent.invoke(initial_state)
    return result
