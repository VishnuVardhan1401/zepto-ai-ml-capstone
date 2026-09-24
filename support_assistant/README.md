# Zepto Support Assistant

This module implements a small RAG-based Zepto support assistant.

The required graded baseline uses deterministic mock logic with
MOCK_LLM left at its default value, so no external LLM API key is
required.

The main components are:

- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- LangGraph
- Pydantic
- FastAPI

---

## RAG Architecture

The complete pipeline is:

ingestion -> embedding -> retrieval -> generation

### 1. Ingestion

The Zepto policy corpus contains 8 documents:

- docs/doc_01.txt
- docs/doc_02.txt
- docs/doc_03.txt
- docs/doc_04.txt
- docs/doc_05.txt
- docs/doc_06.txt
- docs/doc_07.txt
- docs/doc_08.txt

The `build_index()` function in `retrieval.py` loads all 8 documents.

Because the documents are short, each document is treated as one chunk.

Each chunk receives a unique ID such as:

`doc_01_chunk_01`

---

### 2. Embedding

The `retrieval.py` module uses the required local embedding model:

`all-MiniLM-L6-v2`

from the Sentence Transformers library.

Each document chunk is converted into an embedding locally.

No external embedding API is required.

The same embedding model is also used to embed the user's query during
retrieval.

---

### 3. ChromaDB Storage

The generated embeddings are stored in a persistent ChromaDB collection
named:

`zepto_policy`

The collection stores:

- chunk IDs
- document text
- metadata
- embeddings

The persistent ChromaDB data is stored locally in the project's:

`chroma_db/`

directory.

---

### 4. Retrieval

For a policy question, the LangGraph node:

`retrieve_and_answer`

calls the:

`retrieve()`

function from `retrieval.py`.

The user query is embedded using:

`all-MiniLM-L6-v2`

The query embedding is sent to the `zepto_policy` ChromaDB collection.

The top 3 most similar chunks are retrieved using cosine similarity.

The retrieved chunk IDs are stored in the final `sources` field.

The single most similar chunk is also used to construct the mock answer.

---

### 5. Generation

The LangGraph workflow contains three required nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

The `classify_intent` node determines whether a query is:

`policy_question`

or:

`general_question`

A policy question is routed to:

`retrieve_and_answer`

A general question is routed to:

`direct_answer`

The graph uses a conditional edge from `classify_intent` to the appropriate
answer node.

---

## LangGraph Data Flow

The data flow is:

User query
-> FastAPI `/ask`
-> LangGraph `classify_intent`
-> policy question or general question

For a policy question:

classify_intent
-> retrieve_and_answer
-> embed query
-> ChromaDB top-3 retrieval
-> answer generation
-> Pydantic validation
-> FastAPI response

For a general question:

classify_intent
-> direct_answer
-> fixed mock response
-> Pydantic validation
-> FastAPI response

---

## MOCK_LLM Behavior

The environment variable controlling the LLM behavior is:

`MOCK_LLM`

### Default / graded mode

When `MOCK_LLM` is unset or set to:

`MOCK_LLM=1`

the deterministic mock path is used.

No external LLM call is made.

### classify_intent in mock mode

The query is classified using the required keyword heuristic.

The policy keywords are:

- delivery
- return
- refund
- membership
- tracking
- cancel
- gift card
- support hours

If the lowercased query contains one of these keywords, the query is
classified as `policy_question`.

Otherwise it is classified as `general_question`.

### retrieve_and_answer in mock mode

For a `policy_question`, retrieval still runs normally.

The query is embedded locally and ChromaDB returns the top 3 chunks.

The answer is generated deterministically using the format:

`Based on the retrieved context: <top chunk snippet>`

The `sources` field contains the retrieved chunk IDs.

The `confidence` value is set deterministically to:

`1.0`

### direct_answer in mock mode

For a `general_question`, no retrieval is performed.

The deterministic response is:

`I can only answer questions about Zepto policies right now.`

The `sources` list is:

`[]`

The `confidence` value is:

`1.0`

---

## Optional Real-LLM Mode

When:

`MOCK_LLM=0`

the optional real-LLM path is enabled.

The retrieval and embedding stages remain local.

The `classify_intent` generation step can use the real LLM.

The `retrieve_and_answer` node retrieves the top 3 chunks first and then
uses the structured prompt from `prompt.py` to generate a grounded answer.

The `direct_answer` node can prompt the real LLM directly without retrieval.

If the real LLM response fails Pydantic validation, the system retries with a
corrective instruction up to two additional times.

The real-LLM path is optional and is not required for the graded baseline.

---

## Structured Prompt

The structured prompt is implemented in:

`prompt.py`

The prompt follows the required:

Role -> Context -> Task -> Format -> Length

structure.

It also contains:

- an explicit negative constraint preventing answers from using information
  outside the supplied context
- a few-shot example
- the required JSON response format

The prompt is used by the optional real-LLM path.

---

## Pydantic Response Schema

The final response is validated using Pydantic.

The response fields are:

```json
{
  "answer": "string",
  "sources": ["chunk_id"],
  "confidence": 1.0
}
---
##FastAPI

The FastAPI application is implemented in:

main.py

The application exposes:

POST /ask

The request model is:

{
  "query": "string"
}

The response uses the validated Pydantic response model described above.

---
##Running FastAPI

From the project root, run:

python3 -m uvicorn support_assistant.main:app --reload

The application runs at:

http://127.0.0.1:8000

The interactive API documentation is available at:

http://127.0.0.1:8000/docs

----
FastAPI Example 1 - Policy Question

Request:

curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"query":"What is the delivery fee for orders below INR 149?"}'

Response:

{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01_chunk_01",
    "doc_05_chunk_01",
    "doc_03_chunk_01"
  ],
  "confidence": 1
}

-----
FastAPI Example 2 - General Question

Request:

curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"query":"What is the capital of India?"}'

Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1
}
---
Docker

The Dockerfile is located at:

support_assistant/Dockerfile

The Docker image is built from the project root with:

docker build -f support_assistant/Dockerfile -t zepto-support-assistant .

The container is run with:

docker run --rm -p 7861:7860 zepto-support-assistant

The FastAPI application can then be accessed locally at:

http://127.0.0.1:7861

The Docker container serves:

POST /ask

The Docker image uses the required mock baseline:

MOCK_LLM=1

The local Docker build and run are the required graded containerization
baseline.

## Module 3 Completion

The module includes:

8 Zepto policy documents
local embeddings using all-MiniLM-L6-v2
ChromaDB vector storage and retrieval
structured prompt template
LangGraph StateGraph
classify_intent
retrieve_and_answer
direct_answer
conditional graph routing
Pydantic response validation
deterministic mock mode
optional real-LLM path
FastAPI /ask
local Docker deployment