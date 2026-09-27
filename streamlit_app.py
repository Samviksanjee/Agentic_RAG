"""Streamlit interface for the Agentic RAG LangGraph workflow."""

from __future__ import annotations

import time

import streamlit as st

from rag_service import AgentConfigError, runner


st.set_page_config(page_title="Agentic RAG · Research console", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #07111f; }
    [data-testid="stSidebar"] { background: #081525; border-right: 1px solid #1d2b41; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -.04em; margin: 0 0 2.6rem .35rem; }
    .brand span { color: #43d5e5; }
    .eyebrow { color: #6f829e; font: .65rem monospace; letter-spacing: .14em; }
    h1, h2, h3 { letter-spacing: -.04em; }
    h1 { font-size: 2.2rem !important; margin-top: .25rem; }
    .hero { padding: 3rem 0 1.5rem; max-width: 720px; }
    .hero h1 { font-size: clamp(2.2rem, 5vw, 4.1rem) !important; line-height: 1.05; }
    .hero em { color: #43d5e5; font-style: normal; }
    .hero p { color: #8a9bb4; line-height: 1.6; max-width: 590px; }
    .status-card { background: #0d1b2e; border: 1px solid #1d2b41; border-radius: .7rem; padding: .9rem; margin-top: 1.2rem; }
    .status-row { display:flex; justify-content:space-between; color:#7d8ca6; font-size:.78rem; padding:.25rem 0; }
    .ready { color:#66d7a0; } .missing { color:#e48787; }
    .trace-card { background:#0d1b2e; border:1px solid #1d2b41; border-radius:.7rem; padding:.8rem 1rem; margin:.35rem 0; }
    .trace-card small { color:#7386a1; }
    .source-card { background:#102438; border:1px solid #285268; border-radius:.5rem; padding:.65rem .8rem; margin:.35rem 0; }
    .stChatMessage { border: 1px solid #1c2e45; background: #0d1a2c; }
    [data-testid="stChatInput"] { border-color: #315168; }
    .muted { color:#7d8ca6; }
    </style>
    """,
    unsafe_allow_html=True,
)


def service_status(configured: bool) -> str:
    return "<span class='ready'>ready</span>" if configured else "<span class='missing'>missing</span>"


def render_trace(trace: list[str]) -> None:
    labels = {
        "route_question": ("01", "Route question", "LLM decides private KB or direct answer"),
        "retrieve_kb": ("02", "Private KB retrieval", "Semantic search across Pinecone"),
        "grade_kb_evidence": ("03", "Grade KB evidence", "Grounding and similarity check"),
        "search_web": ("04", "Tavily web fallback", "Search only when private evidence is weak"),
        "grade_web_evidence": ("05", "Grade web evidence", "Validate external evidence"),
        "rewrite_query": ("06", "Rewrite query", "Retry guard: one improved retrieval attempt"),
        "generate_from_kb": ("07", "Generate from Private KB", "Grounded response synthesis"),
        "generate_from_web": ("07", "Generate from Web", "Web-grounded response synthesis"),
        "direct_answer": ("07", "Direct answer", "Simple conversational response"),
        "answer_insufficient": ("07", "Insufficient evidence", "Transparent limitation response"),
    }
    st.markdown("#### Evidence path")
    if not trace:
        st.caption("No run yet")
    for node in trace:
        number, title, description = labels.get(node, ("·", node, "Workflow node"))
        st.markdown(f"<div class='trace-card'><strong>{number} · {title}</strong><br><small>{description}</small></div>", unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    valid = [source for source in sources if str(source.get("url", "")).startswith(("http://", "https://"))]
    if not valid:
        return
    st.markdown("#### Sources")
    for source in valid[:5]:
        title = source.get("title") or source.get("url")
        st.markdown(f"- [{title}]({source['url']})")


def text_chunks(text: str, size: int = 72):
    """Yield small text chunks so Streamlit can render the answer progressively."""
    for start in range(0, len(text), size):
        yield text[start : start + size]
        time.sleep(0.01)


def offline_fallback(question: str) -> str:
    """Give a transparent local explanation when provider networking is unavailable."""
    lowered = question.lower()
    if "relevant" in lowered or "retriev" in lowered or "weak" in lowered:
        return (
            "The configured provider connection is unavailable, so I could not run the live graph. "
            "From this project's architecture: when retrieved private-KB documents are not relevant, "
            "the evidence grader marks them weak, the workflow falls back to Tavily web search, grades "
            "that evidence, and can rewrite the query once before returning an insufficient-evidence message."
        )
    if "architect" in lowered or "flow" in lowered:
        return (
            "The live provider connection is unavailable, but the configured flow is: route the question, "
            "retrieve from the private Pinecone KB, grade evidence, generate from the KB when it passes, "
            "or search Tavily and grade web evidence when it does not."
        )
    return (
        "I received your question, but the configured Groq connection is blocked in this environment. "
        "The Streamlit UI is working; allow outbound access to the configured providers and retry for a live, grounded answer."
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "section" not in st.session_state:
    st.session_state.section = "Ask the agent"


with st.sidebar:
    st.markdown("<div class='brand'>✦ agentic<span>.rag</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>WORKSPACE</div>", unsafe_allow_html=True)
    st.caption("Research console")
    st.session_state.section = st.radio("Navigate", ["Ask the agent", "Knowledge sources", "Agent flow"], label_visibility="collapsed")
    if st.button("↺  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    status = runner.status()
    st.markdown("<div class='status-card'><strong>System status</strong>", unsafe_allow_html=True)
    for key, label in (("groq", "Groq LLM"), ("pinecone", "Pinecone KB"), ("tavily", "Tavily search")):
        st.markdown(f"<div class='status-row'><span>{label}</span>{service_status(status['services'][key])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Credentials stay server-side in .env")


if st.session_state.section == "Knowledge sources":
    st.markdown("<div class='eyebrow'>KNOWLEDGE SOURCES</div>", unsafe_allow_html=True)
    st.title("Private knowledge base")
    st.write("The agent attempts trusted private evidence first. Web search is used only when the KB evidence is graded weak.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vector index", status["index"])
        st.metric("Namespace", status["namespace"])
    with col2:
        st.metric("Top K", "4")
        st.metric("Similarity threshold", "0.65")
    st.info("The configured Pinecone index must already exist. The UI does not upload, delete, or mutate documents.")
    st.markdown("#### Ingestion architecture")
    st.code("Documents → RecursiveCharacterTextSplitter → all-MiniLM-L6-v2 → Pinecone\n\nQuestion → Router → Private KB → Evidence grader → Answer / Tavily fallback", language="text")
elif st.session_state.section == "Agent flow":
    st.markdown("<div class='eyebrow'>LANGGRAPH ORCHESTRATION</div>", unsafe_allow_html=True)
    st.title("Agent flow")
    st.write("Every decision follows the architecture described in the project README.")
    render_trace(["route_question", "retrieve_kb", "grade_kb_evidence", "search_web", "grade_web_evidence", "rewrite_query", "generate_from_kb"])
    st.warning("The web fallback and rewrite branch are conditional: they run only when private evidence is insufficient.")
else:
    st.markdown("<div class='hero'><div class='eyebrow'>AGENTIC RESEARCH · LANGGRAPH ORCHESTRATED</div><h1>What can I help you <em>find?</em></h1><p>Ask about your Agentic RAG system. I’ll check the private knowledge base first, verify the evidence, then use web search only when needed.</p></div>", unsafe_allow_html=True)
    if not st.session_state.messages:
        st.info("Try: What happens when retrieved documents are not relevant?")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                st.caption(f"Source: {message['source']}")
                with st.expander("View evidence path"):
                    render_trace(message.get("trace", []))
                    render_sources(message.get("sources", []))
    prompt = st.chat_input("Ask a question about Agentic RAG…", max_chars=2000)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            result = None
            fallback = None
            with st.status("Tracing the best evidence path…", expanded=False) as status_box:
                try:
                    result = runner.ask(" ".join(prompt.split()))
                    status_box.update(label="Evidence path complete", state="complete")
                except AgentConfigError as exc:
                    status_box.update(label="Configuration required", state="error")
                    st.error(str(exc))
                except Exception as exc:
                    error_text = str(exc).lower()
                    if "connection" in error_text or "10013" in error_text or "connecterror" in error_text:
                        fallback = offline_fallback(prompt)
                        status_box.update(label="Offline explanation · answer streamed", state="complete")
                    else:
                        status_box.update(label="Agent unavailable", state="error")
                        st.error("The agent could not complete this request. Check dependencies, API keys, and the Pinecone index.")
            if result:
                st.write_stream(text_chunks(result["answer"]))
                st.caption(f"Source: {result['source']}")
                with st.expander("View evidence path"):
                    render_trace(result.get("trace", []))
                    render_sources(result.get("sources", []))
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "source": result["source"], "trace": result.get("trace", []), "sources": result.get("sources", [])})
            elif fallback:
                st.write_stream(text_chunks(fallback))
                st.caption("Source: offline explanation (live providers unavailable)")
                st.session_state.messages.append({"role": "assistant", "content": fallback, "source": "offline_fallback", "trace": ["route_question", "offline_fallback"], "sources": []})
