# Industry-Style Agentic RAG with LangGraph

An industry-style **Agentic Retrieval-Augmented Generation (RAG)** system orchestrated with **LangGraph**.

The system is designed around a simple principle:

> **Use the private knowledge base first, verify the retrieved evidence, and use web search only when the private evidence is insufficient.**

It combines:

- **LangGraph** for stateful agentic orchestration
- **Pinecone** for private vector storage and semantic retrieval
- **Hugging Face Sentence Transformers** for local embeddings
- **Groq** for LLM inference
- **Tavily** for live web-search fallback
- **LangChain** for the RAG components
- **Structured LLM decisions** for routing and evidence grading

---

## Architecture

![Agentic RAG Architecture](architecture.png)

The architecture follows this high-level flow:

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  LLM Router      │
                         │                  │
                         │ KB / Direct      │
                         └───────┬──────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                  KB path                 Direct path
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐        ┌──────────────────┐
          │ Pinecone Private │        │ Direct LLM       │
          │ Knowledge Base   │        │ Answer           │
          └────────┬─────────┘        └────────┬─────────┘
                   │                           │
                   ▼                           │
          ┌──────────────────┐                 │
          │ Grade KB Evidence│                 │
          │ LLM + similarity │                 │
          └────────┬─────────┘                 │
                   │                           │
             ┌─────┴─────┐                     │
             │           │                     │
           GOOD         WEAK                    │
             │           │                     │
             ▼           ▼                     │
       ┌───────────┐ ┌──────────────┐           │
       │ Generate  │ │ Tavily Web   │           │
       │ from KB   │ │ Search       │           │
       └─────┬─────┘ └──────┬───────┘           │
             │              │                   │
             │              ▼                   │
             │       ┌──────────────┐            │
             │       │ Grade Web    │            │
             │       │ Evidence     │            │
             │       └──────┬───────┘            │
             │              │                   │
             │         ┌────┴────┐              │
             │        GOOD      WEAK             │
             │          │         │              │
             │          ▼         ▼              │
             │      Web Answer  Rewrite Query    │
             │                    │              │
             │                    └──► KB        │
             │                                   │
             └───────────────────┬───────────────┘
                                 ▼
                         ┌──────────────────┐
                         │ Final Answer     │
                         │ + Source Type    │
                         └──────────────────┘
```

---

## What makes this Agentic RAG?

A basic RAG pipeline normally follows:

```text
Question → Retrieve → Generate
```

This project adds decision-making and recovery:

```text
Question
   ↓
Route
   ↓
Private KB Retrieval
   ↓
Grade Evidence
   ├── Good → Generate from Private KB
   │
   └── Weak → Web Search
                 ↓
            Grade Web Evidence
                 ├── Good → Generate from Web
                 │
                 └── Weak → Rewrite Query
                                ↓
                         Retrieve from KB again
```

The workflow is controlled by **LangGraph**, so every decision is represented as a graph node or conditional edge.

---

## Knowledge Base

For the notebook demo, the private knowledge base is created from the LangGraph Agentic RAG documentation:

**Source document**

```text
https://docs.langchain.com/oss/python/langgraph/agentic-rag
```

The document is:

```text
Web document
     ↓
BeautifulSoup / WebBaseLoader
     ↓
RecursiveCharacterTextSplitter
     ↓
Hugging Face embeddings
     ↓
Pinecone
```

### Chunking configuration

The notebook uses:

```python
chunk_size = 1000
chunk_overlap = 150
```

### Embedding model

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting vectors have **384 dimensions**.

### Pinecone configuration

```text
Index:
industry-agentic-rag-kb

Namespace:
langgraph-agentic-rag

Metric:
cosine

Dimension:
384

Cloud:
AWS

Region:
us-east-1
```

---

## Agentic Workflow

### 1. Question Router

The router uses an LLM with structured output to decide whether the question should use:

```text
kb
```

or:

```text
direct
```

The KB route is used for questions related to:

- Agentic RAG
- LangGraph Agentic RAG
- retrieval grading
- query rewriting
- RAG architecture
- retriever tools
- web fallback

Simple greetings and basic conversation can use the direct route.

---

### 2. Private KB Retrieval

The selected query is sent to the Pinecone-backed retriever.

The notebook retrieves:

```text
Top K = 4
```

documents/chunks.

The important design principle is:

> **Private KB first.**

The system does not immediately send every question to Tavily.

---

### 3. Private Evidence Grading

The retrieved private evidence is evaluated by an LLM.

The grader returns:

```text
good
```

or:

```text
weak
```

The notebook also checks vector similarity when scores are available.

Current threshold:

```python
SIMILARITY_THRESHOLD = 0.65
```

Minimum relevant chunks:

```python
MIN_RELEVANT_CHUNKS = 1
```

The final KB decision requires:

```text
LLM evidence = good
AND
vector similarity passes
AND
minimum evidence requirement passes
```

This helps prevent the system from blindly trusting arbitrary retrieved chunks.

---

### 4. Generate from Private KB

When the private evidence is sufficient, the answer is generated using only the retrieved Private KB context.

The generation prompt explicitly instructs the LLM:

- Do not use outside knowledge
- Do not invent missing information
- Use only information supported by the Private KB
- Explain the answer clearly
- Mention that the answer is based on the Private KB
- Include the source when possible

The returned source label is:

```text
private_kb
```

---

### 5. Tavily Web Fallback

If the private evidence is weak, the current notebook activates Tavily.

Tavily is configured with:

```python
max_results=5
topic="general"
include_answer=True
include_raw_content=False
```

The web search result is converted into readable text containing:

- Tavily answer, when available
- result title
- URL
- result content

The returned source label is:

```text
web
```

---

### 6. Web Evidence Grading

The web results are also graded before generation.

The web evidence grader returns:

```text
good
```

or:

```text
weak
```

If the web evidence is good, the system generates a web-grounded answer.

If it is weak, the system can rewrite the query.

---

### 7. Query Rewriting

The query rewriter improves the question for retrieval and web search.

It is instructed to:

- preserve the original intent
- make the query more specific
- make it search-friendly
- avoid answering the question

The rewritten query is then sent back through the KB retrieval stage.

A retry guard prevents an infinite loop.

Current setting:

```python
MAX_RETRIES = 1
```

---

### 8. Insufficient Evidence

If the web evidence remains insufficient after the allowed retry, the system returns a transparent insufficient-evidence response rather than pretending to know the answer.

Source label:

```text
insufficient_evidence
```

---

## LangGraph State

The workflow uses an `AgentState` containing:

```python
class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
```

This shared state allows each node to read information produced by previous nodes and update the workflow.

---

## Graph Nodes

| Node | Purpose |
|---|---|
| `route_question` | Decide KB vs direct answer |
| `retrieve_kb` | Retrieve private documents from Pinecone |
| `grade_kb_evidence` | Validate private KB evidence |
| `search_web` | Search Tavily when KB evidence is weak |
| `grade_web_evidence` | Validate web evidence |
| `rewrite_query` | Improve a weak query |
| `generate_from_kb` | Generate a private-KB-grounded answer |
| `generate_from_web` | Generate a web-grounded answer |
| `direct_answer` | Handle simple conversational requests |
| `answer_insufficient` | Return a transparent insufficient-evidence response |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM framework | LangChain |
| LLM | Groq |
| LLM model | `openai/gpt-oss-20b` |
| Private Vector DB | Pinecone |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Web Search | Tavily |
| Web Loading | LangChain `WebBaseLoader` |
| HTML Parsing | BeautifulSoup |
| Text Splitting | RecursiveCharacterTextSplitter |
| Environment Variables | python-dotenv |
| Notebook | Jupyter / VS Code |

---

## Project Structure

A recommended repository structure is:

```text
agentic-rag/
│
├── Agentic_RAG.ipynb
├── README.md
├── architecture.png
├── .env
└── .gitignore
```

### Important

Never commit `.env` to Git.

Example `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.ipynb_checkpoints/
```

---

## API Keys

The notebook uses three API keys:

```text
GROQ_API_KEY
TAVILY_API_KEY
PINECONE_API_KEY
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
```

The notebook also supports entering missing keys interactively with `getpass`.

**Do not commit real API keys to GitHub.**

If a key is accidentally exposed, rotate/revoke it from the corresponding provider.

---

## Installation

The notebook installs the main dependencies with:

```bash
pip install langgraph==1.2.11 \
            langchain==1.3.16 \
            langchain-groq==1.1.3 \
            langchain-pinecone==0.2.13 \
            langchain-tavily==0.2.18 \
            pinecone==7.3.0
```

Additional dependencies used by the notebook include:

```bash
pip install python-dotenv
pip install langchain-community
pip install beautifulsoup4==4.15.0
pip install langchain-huggingface==1.2.2
pip install sentence-transformers==6.0.0
```

---

## Running the Notebook

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd agentic-rag
```

### 2. Create and activate a virtual environment

For example:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

Install the packages listed above.

### 4. Configure `.env`

Add:

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
PINECONE_API_KEY=...
```

### 5. Open the notebook

Open:

```text
Agentic_RAG.ipynb
```

in Jupyter Notebook, JupyterLab, or VS Code.

### 6. Run cells in order

The notebook is organized into the following stages:

```text
1. Install Dependencies
2. Configure API Keys
3. Load Real Documents
4. Split Documents
5. Create Embeddings
6. Create / Load Pinecone
7. Test Private KB Retrieval
8. Initialize Groq
9. Initialize Tavily
10. Define Structured Decisions
11. Route Question
12. Retrieve Private KB
13. Grade KB Evidence
14. Tavily Search
15. Grade Web Evidence
16. Rewrite Query
17. Generate from Private KB
18. Generate from Web
19. Direct Answer
20. Insufficient Evidence
21. Build LangGraph
22. Visualize Graph
23. Helper Function
24-27. Run Demo Questions
28. Review Industry-Ready Features
```

---

## Running the Agent

The notebook exposes a helper function:

```python
ask_agent(question)
```

Example:

```python
result = ask_agent(
    "In Agentic RAG, what happens when retrieved documents are not relevant?"
)
```

The helper prints:

```text
QUESTION:
...

SOURCE USED:
...

FINAL ANSWER:
...
```

---

## Demo Scenarios

### Demo 1 — Private KB

```python
result = ask_agent(
    "In Agentic RAG, what happens when retrieved documents are not relevant?"
)
```

Expected path:

```text
Question
   ↓
Router
   ↓
Private KB
   ↓
KB Evidence Grader
   ↓
Generate from Private KB
```

---

### Demo 2 — Web Fallback

```python
result = ask_agent(
    "What is Tavily Search and why is it useful for AI agents and RAG workflows?"
)
```

This demonstrates the fallback path when the private knowledge base is not sufficient.

```text
Question
   ↓
Private KB
   ↓
Evidence weak
   ↓
Tavily
   ↓
Web Evidence Grader
   ↓
Generate from Web
```

---

### Demo 3 — Direct Answer

```python
result = ask_agent(
    "Hello, how are you?"
)
```

Expected path:

```text
Question
   ↓
Router
   ↓
Direct Answer
```

No vector retrieval is required.

---

### Demo 4 — Current / External Question

```python
result = ask_agent(
    "What is the current LangChain Tavily package used for Python web search integration?"
)
```

This demonstrates how the system can use the web when the private KB does not contain sufficient information.

---

## Private KB vs Hybrid Mode

The current notebook demonstrates a **hybrid Agentic RAG** design:

```text
Private KB
    ↓
Evidence grading
    ↓
Weak?
    ↓
Tavily
```

For a strict private-KB deployment, the same architecture can be extended with a configuration switch such as:

```python
ALLOW_WEB_FALLBACK = False
```

Then:

```text
KB good
   ↓
Answer from Private KB

KB weak
   ↓
Web fallback disabled
   ↓
Information not available in Private KB
```

With:

```python
ALLOW_WEB_FALLBACK = True
```

the system can operate as:

```text
KB good
   ↓
Answer from Private KB

KB weak
   ↓
Tavily Web Search
   ↓
Grade Web Evidence
   ↓
Web-grounded answer
```

This makes the fallback behavior configurable for different deployment scenarios.

---

## Source Tracking

Every final response records where the answer came from.

Possible values include:

```text
private_kb
web_search
direct
insufficient_evidence
```

This is useful for:

- debugging
- observability
- user transparency
- evaluating retrieval quality
- building UI source indicators
- enterprise audit workflows

---

## Why This Is More Industry-Ready

The notebook demonstrates several patterns found in production-oriented RAG systems:

### 1. Private KB first

The system attempts to use trusted internal information before relying on external sources.

### 2. Evidence grading

Retrieved information is evaluated before generation.

### 3. Hybrid retrieval

The system can combine private retrieval with live web search.

### 4. Query rewriting

Weak retrieval can trigger a better query.

### 5. Retry protection

The retry counter prevents uncontrolled loops.

### 6. Source tracking

The application knows whether the answer came from:

```text
Private KB
Web Search
Direct LLM
Insufficient Evidence
```

### 7. Stateful orchestration

LangGraph makes the workflow explicit and debuggable instead of hiding the entire process inside one function.

---

## Important Design Considerations

### Similarity threshold

The notebook starts with:

```python
SIMILARITY_THRESHOLD = 0.65
```

This is a tuning parameter, not a universal value.

The appropriate threshold depends on:

- embedding model
- document quality
- chunk size
- chunk overlap
- domain
- question distribution
- Pinecone similarity scores

For a production system, evaluate the threshold against a representative test set.

### Embedding dimension

Because:

```text
all-MiniLM-L6-v2 → 384 dimensions
```

the Pinecone index is created with:

```python
dimension=384
```

If the embedding model changes, the Pinecone index dimension must match the new embedding model.

### Private data

The notebook currently uses a public LangGraph documentation page as the demonstration source for the private KB.

For an enterprise deployment, replace this with authorized internal documents such as:

```text
PDF
DOCX
TXT
Markdown
CSV
Company documentation
Product manuals
Internal policies
```

and apply appropriate access controls.

---

## Future Extensions

This notebook can be extended into a production application with:

- production API boundary for multi-client deployments
- Streamlit UI (current) or React frontend (future)
- authentication and role-based access
- document upload pipeline
- background ingestion jobs
- metadata-based filtering
- multi-tenant namespaces
- citation rendering
- conversation memory
- evaluation datasets
- LangSmith tracing
- structured observability
- human approval workflows
- multiple private data sources
- SQL/database tools
- enterprise document connectors
- configurable web fallback
- response caching
- cost monitoring

A possible production architecture is:

```text
                    ┌──────────────────────┐
                    │  Streamlit Frontend  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Agent Runner      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      LangGraph       │
                    │   Agent Controller   │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Private KB          Web Search        Direct LLM
       Pinecone             Tavily
             │                 │
             └────────┬────────┘
                      ▼
               Evidence Grading
                      │
                      ▼
               Grounded Answer
                      │
                      ▼
                  Frontend
```

---

## Evaluation Ideas

For a stronger project evaluation, create a test set containing:

```text
1. Questions answerable directly from the private KB
2. Questions partially answerable from the private KB
3. Questions not covered by the private KB
4. Current questions requiring web search
5. Ambiguous questions
6. Simple conversational questions
```

Measure:

- KB retrieval accuracy
- evidence grading accuracy
- web fallback rate
- answer groundedness
- citation/source correctness
- query rewrite success rate
- latency
- LLM/API cost

---

## Troubleshooting

### Pinecone dimension error

Verify that:

```text
Embedding dimension = 384
Pinecone index dimension = 384
```

### Groq authentication error

Check:

```python
import os
print(bool(os.getenv("GROQ_API_KEY")))
```

Also verify that the notebook is running in the intended Python environment.

Never print the full API key.

### Tavily authentication error

Check:

```python
import os
print(bool(os.getenv("TAVILY_API_KEY")))
```

### Pinecone authentication error

Check:

```python
import os
print(bool(os.getenv("PINECONE_API_KEY")))
```

### Notebook uses the wrong Python environment

Check:

```python
import sys
print(sys.executable)
```

The notebook should use the Python environment where the project dependencies were installed.

### Pinecone index already exists

The notebook checks whether the configured index exists before creating it. If it already exists, the existing index can be loaded.

---

## Security

Do not commit credentials.

Use:

```text
.env
```

and add it to:

```text
.gitignore
```

Never hard-code:

```text
GROQ_API_KEY
TAVILY_API_KEY
PINECONE_API_KEY
```

into source code committed to a public repository.

---

## License

Add the license appropriate for your project before publishing the repository.

---

## Summary

This project demonstrates an **Agentic RAG workflow rather than a simple retrieve-and-generate pipeline**.

The core idea is:

```text
                 ┌────────────────────┐
                 │      Question      │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │   Route Question   │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │   Private KB       │
                 │     Pinecone       │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │  Grade Evidence    │
                 └─────────┬──────────┘
                           ↓
                    ┌──────┴──────┐
                    │             │
                  GOOD           WEAK
                    │             │
                    ↓             ↓
               Private KB      Tavily
                 Answer         Search
                                  ↓
                           Grade Evidence
                                  ↓
                           Generate / Retry
```

The result is a transparent, stateful RAG workflow that can inspect evidence, make routing decisions, recover from weak retrieval, and identify the source used for the final answer.

## Streamlit UI

The repository includes a single Streamlit application in `streamlit_app.py` and a lazy LangGraph runner in `rag_service.py`. Keeping the UI and runner in one Streamlit process makes the chat, knowledge-source view, and agent-flow view work from the same session state.

The interface includes:

- private-first chat with example prompts
- live route observability for router, retrieval, grading, fallback, and generation
- source badges for Private KB, Web Search, Direct LLM, or insufficient evidence
- citation links for retrieved KB and Tavily sources
- responsive layout for desktop and mobile widths
- a sidebar health panel showing whether Groq, Pinecone, and Tavily are configured
- dedicated Knowledge Sources and Agent Flow views
- streamed answer rendering with a transparent offline explanation when provider networking is blocked

### Run locally

Install the pinned project dependencies, including Streamlit:

```powershell
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the provider keys. The server expects the existing Pinecone index `industry-agentic-rag-kb` in the `langgraph-agentic-rag` namespace; it does not upload or mutate documents when the UI starts.

Run it from the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

Then open the URL shown by Streamlit, usually `http://localhost:8501`. Credentials remain server-side in `.env`; external services are initialized only when the first question is submitted.

The app expects the existing Pinecone index `industry-agentic-rag-kb` in the `langgraph-agentic-rag` namespace. It does not upload, delete, or mutate documents when the UI starts.

Questions are limited to 2,000 characters and normalized before execution. Provider errors are shown as a generic configuration message in the UI so credentials and internal stack traces are not exposed.

The UI can still load and navigate without provider access. Answer generation requires outbound connectivity to the configured Groq, Pinecone, Hugging Face, and (when needed) Tavily services; a blocked network returns an actionable connection message.
