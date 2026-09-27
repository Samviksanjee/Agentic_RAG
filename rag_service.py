"""A lazy, API-friendly LangGraph runner for the notebook architecture."""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv


class AgentConfigError(RuntimeError):
    """Raised when the configured RAG services cannot be used."""


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: list[Any]
    web_results: str
    web_sources: list[dict[str, str]]
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: list[str]


class AgentRunner:
    """Builds the graph only when the first question is submitted."""

    index_name = "industry-agentic-rag-kb"
    namespace = "langgraph-agentic-rag"
    max_retries = 1
    similarity_threshold = 0.65

    def __init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        self._graph = None
        self._retriever = None
        self._vectorstore = None
        self._llm = None
        self._router_llm = None
        self._kb_grader_llm = None
        self._web_grader_llm = None
        self._web_search = None
        self._kb_ready = False

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._graph is not None,
            "services": {
                "groq": bool(os.getenv("GROQ_API_KEY")),
                "pinecone": bool(os.getenv("PINECONE_API_KEY")),
                "tavily": bool(os.getenv("TAVILY_API_KEY")),
            },
            "index": self.index_name,
            "namespace": self.namespace,
        }

    @staticmethod
    def _structured_value(response: Any, key: str, allowed: set[str]) -> str | None:
        """Read provider JSON consistently across model/schema response variants."""
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        text = str(content).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        data: dict[str, Any] = {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            match = re.search(r"\b(route|grade|result)\s*['\"]?\s*:\s*['\"](kb|direct|good|weak)['\"]", text, re.IGNORECASE)
            if match:
                data = {match.group(1).lower(): match.group(2).lower()}
        value = data.get(key) or data.get("result")
        value = str(value).lower() if value is not None else ""
        return value if value in allowed else None

    def _init_kb(self) -> None:
        if self._kb_ready:
            return
        if not os.getenv("PINECONE_API_KEY"):
            raise AgentConfigError("Missing required environment variable: PINECONE_API_KEY")
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing = pinecone.list_indexes()
        names = {item["name"] if isinstance(item, dict) else getattr(item, "name", "") for item in existing}
        if self.index_name not in names:
            raise AgentConfigError(f"Pinecone index '{self.index_name}' was not found")
        index = pinecone.Index(self.index_name)
        self._vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=self.namespace)
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4, "namespace": self.namespace})
        self._kb_ready = True

    def _init(self) -> None:
        if self._graph is not None:
            return
        missing = [key for key in ("GROQ_API_KEY",) if not os.getenv(key)]
        if missing:
            raise AgentConfigError(f"Missing required environment variables: {', '.join(missing)}")

        from langchain_groq import ChatGroq
        from langchain_tavily import TavilySearch
        from pydantic import BaseModel, Field
        from langgraph.graph import END, START, StateGraph

        class RouteDecision(BaseModel):
            route: Literal["kb", "direct"] = Field(description="KB for RAG questions, direct for simple conversation")

        class EvidenceGrade(BaseModel):
            grade: Literal["good", "weak"] = Field(description="Whether the evidence supports an answer")

        self._llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        self._router_llm = self._llm.with_structured_output(RouteDecision, method="json_mode")
        self._kb_grader_llm = self._llm.with_structured_output(EvidenceGrade, method="json_mode")
        self._web_grader_llm = self._llm.with_structured_output(EvidenceGrade, method="json_mode")
        self._web_search = (
            TavilySearch(max_results=5, topic="general", include_answer=True, include_raw_content=False)
            if os.getenv("TAVILY_API_KEY")
            else None
        )

        def add_trace(state: AgentState, label: str) -> list[str]:
            return [*state.get("trace", []), label]

        def route_question(state: AgentState):
            decision = self._llm.invoke(
                """Route to kb for questions about Agentic RAG, LangGraph workflows, retrieval grading, query rewriting, RAG architecture, retrievers, or web fallback. Route to direct only for greetings, thanks, or simple conversation. Return JSON.\nQuestion: """ + state["question"]
            )
            route = self._structured_value(decision, "route", {"kb", "direct"}) or "kb"
            return {"current_query": state["question"], "source_used": route, "trace": add_trace(state, "route_question")}

        def after_route(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
            return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

        def retrieve_kb(state: AgentState):
            self._init_kb()
            docs = self._vectorstore.similarity_search_with_score(state["current_query"], k=4, namespace=self.namespace)
            normalized = []
            for doc, score in docs:
                doc.metadata["score"] = float(score)
                normalized.append(doc)
            return {"kb_docs": normalized, "trace": add_trace(state, "retrieve_kb")}

        def grade_kb(state: AgentState):
            docs = state["kb_docs"]
            context = "\n\n".join(f"[PRIVATE KB SOURCE: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs)
            grade_response = self._llm.invoke(
                "Determine whether the private evidence directly supports the question. Return good only when it does; otherwise weak. Do not use outside knowledge. Return JSON.\nQuestion: "
                + state["question"]
                + "\nEvidence:\n"
                + context
            )
            grade = self._structured_value(grade_response, "grade", {"good", "weak"}) or "weak"
            scores = [float(doc.metadata["score"]) for doc in docs if doc.metadata.get("score") is not None]
            max_score = max(scores) if scores else None
            passed = grade == "good" and (max_score is None or max_score >= self.similarity_threshold) and bool(docs)
            return {"kb_grade": "good" if passed else "weak", "trace": add_trace(state, "grade_kb_evidence")}

        def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
            return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

        def search_web(state: AgentState):
            if self._web_search is None:
                return {"web_results": "", "web_sources": [], "source_used": "web", "trace": add_trace(state, "search_web")}
            result = self._web_search.invoke({"query": state["current_query"]})
            answer = result.get("answer", "") if isinstance(result, dict) else ""
            raw_sources = result.get("results", []) if isinstance(result, dict) else []
            sources = [{"title": str(item.get("title", "Source")), "url": str(item.get("url", ""))} for item in raw_sources if item.get("url")]
            body = ([f"Tavily answer: {answer}"] if answer else []) + [f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nContent: {item.get('content', '')}" for item in raw_sources]
            return {"web_results": "\n\n".join(body), "web_sources": sources, "source_used": "web", "trace": add_trace(state, "search_web")}

        def grade_web(state: AgentState):
            if not state["web_results"]:
                grade = "weak"
            else:
                grade_response = self._llm.invoke("Can this web evidence answer the question? Return good if yes, otherwise weak. Return JSON.\nQuestion: " + state["question"] + "\nEvidence:\n" + state["web_results"])
                grade = self._structured_value(grade_response, "grade", {"good", "weak"}) or "weak"
            return {"web_grade": grade, "trace": add_trace(state, "grade_web_evidence")}

        def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
            if state["web_grade"] == "good":
                return "generate_from_web"
            return "rewrite_query" if state["retry_count"] < self.max_retries else "answer_insufficient"

        def rewrite_query(state: AgentState):
            rewritten = self._llm.invoke("Rewrite this question for specific retrieval and web search. Preserve intent. Return only the rewritten query.\n" + state["question"]).content.strip()
            return {"current_query": rewritten, "retry_count": state["retry_count"] + 1, "trace": add_trace(state, "rewrite_query")}

        def generate_kb(state: AgentState):
            context = "\n\n".join(f"[PRIVATE KB SOURCE: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in state["kb_docs"])
            answer = self._llm.invoke("Answer using ONLY the private KB context. Be beginner-friendly, do not invent details, and mention that the answer is based on the Private KB.\nQuestion: " + state["question"] + "\nContext:\n" + context).content
            return {"answer": answer, "source_used": "private_kb", "trace": add_trace(state, "generate_from_kb")}

        def generate_web(state: AgentState):
            answer = self._llm.invoke("Answer using ONLY this web search context. Be beginner-friendly, do not invent details, and mention Tavily web search.\nQuestion: " + state["question"] + "\nContext:\n" + state["web_results"]).content
            return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "generate_from_web")}

        def direct_answer(state: AgentState):
            answer = self._llm.invoke("Respond briefly and naturally.\nMessage: " + state["question"]).content
            return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "direct_answer")}

        def insufficient(state: AgentState):
            return {"answer": "I could not find enough reliable evidence in the private knowledge base or web search to answer this confidently. Please rephrase the question or provide a more specific document.", "source_used": "insufficient_evidence", "trace": add_trace(state, "answer_insufficient")}

        workflow = StateGraph(AgentState)
        for name, fn in (("route_question", route_question), ("retrieve_kb", retrieve_kb), ("grade_kb_evidence", grade_kb), ("search_web", search_web), ("grade_web_evidence", grade_web), ("rewrite_query", rewrite_query), ("generate_from_kb", generate_kb), ("generate_from_web", generate_web), ("direct_answer", direct_answer), ("answer_insufficient", insufficient)):
            workflow.add_node(name, fn)
        workflow.add_edge(START, "route_question")
        workflow.add_conditional_edges("route_question", after_route, {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"})
        workflow.add_edge("retrieve_kb", "grade_kb_evidence")
        workflow.add_conditional_edges("grade_kb_evidence", after_kb, {"generate_from_kb": "generate_from_kb", "search_web": "search_web"})
        workflow.add_edge("search_web", "grade_web_evidence")
        workflow.add_conditional_edges("grade_web_evidence", after_web, {"generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "answer_insufficient": "answer_insufficient"})
        workflow.add_edge("rewrite_query", "retrieve_kb")
        for final in ("generate_from_kb", "generate_from_web", "direct_answer", "answer_insufficient"):
            workflow.add_edge(final, END)
        self._graph = workflow.compile()

    def ask(self, question: str) -> dict[str, Any]:
        self._init()
        initial: AgentState = {"question": question, "current_query": question, "kb_docs": [], "web_results": "", "web_sources": [], "kb_grade": "", "web_grade": "", "answer": "", "source_used": "", "retry_count": 0, "trace": []}
        result = self._graph.invoke(initial)
        sources = [{"title": str(doc.metadata.get("source", "Private KB")), "url": str(doc.metadata.get("source", "")), "score": doc.metadata.get("score")} for doc in result.get("kb_docs", []) if doc.metadata.get("source")]
        return {"answer": result["answer"], "source": result["source_used"], "trace": result.get("trace", []), "sources": sources if result["source_used"] == "private_kb" else result.get("web_sources", []), "diagnostics": {"kb_grade": result.get("kb_grade"), "web_grade": result.get("web_grade"), "retry_count": result.get("retry_count", 0)}}


runner = AgentRunner()


