# Multi-Vector Retriever Demo 

  
This project demonstrates a multi-vector retrieval flow using:

- LangChain + LangGraph concepts
- PGVector as the vector database
- Ollama for embeddings and LLM generation
- A small text document split into chunks and summarized before retrieval

The script in [demo.py](demo.py) loads [test.txt](test.txt), splits it into chunks, summarizes each chunk with Ollama, stores the summaries in PGVector, and then retrieves the matching original parent chunks using `MultiVectorRetriever`.

## Implementation Diagram

```mermaid
flowchart TD
	A[test.txt] --> B[Load with TextLoader]
	B --> C[Split into chunks]

	%% Indexing paths
	C --> D1[Naive path]
	C --> D2[Multi-vector path]

	D1 --> E1[Embed raw chunks<br/>nomic-embed-text]
	E1 --> F1[(PGVector<br/>naive_raw_chunks)]

	D2 --> E2[Summarize chunks<br/>llama3.2:1b]
	E2 --> F2[Summary documents<br/>metadata: doc_id + chunk_index]
	F2 --> G2[Embed summaries<br/>nomic-embed-text]
	G2 --> H2[(PGVector<br/>mv_summaries)]
	C --> I2[(InMemoryStore<br/>parent chunks by doc_id)]

	%% Query-time comparison
	Q[User question] --> N1[Naive similarity_search k=2]
	Q --> N2[MultiVectorRetriever invoke k=2]

	F1 --> N1
	H2 --> N2
	I2 --> N2

	N1 --> O1[Naive top-2 raw chunks]
	N2 --> O2[Top-2 parent chunks via summary->parent mapping]

	O1 --> P[Side-by-side output in demo.py]
	O2 --> P
```

The comparison is performed in a single run of [demo.py](demo.py), which prints:

- naive top-2 raw chunk matches
- summary hits used by the multi-vector path
- multi-vector top-2 parent chunk matches

## Prerequisites

Make sure you have installed:

- Docker Desktop or Docker Engine
- Python 3.9–3.11 (matches the project config)
- Poetry
- Ollama

## Step-by-step setup

### 1) Start the supporting services

This repository includes a Docker Compose file with Postgres + pgvector and a few optional admin/Redis tools.

```bash
docker compose up -d
```

This starts the PostgreSQL instance used by the project on port `6024` and exposes pgAdmin on `http://localhost:5050`.

The project connection string is:

```text
postgresql+psycopg://langchain:langchain@localhost:6024/langchain
```

> The demo uses PGVector, not Redis, for the main retrieval index.

### 2) Install Python dependencies

Using Poetry with your project environment:

```bash
poetry install
```

Then continue with the project commands below.

### 3) Install and prepare the Ollama models

The app uses:

- `nomic-embed-text` for embeddings
- `llama3.2:1b` for summarization

Pull them with:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

Then confirm Ollama is running locally:

```bash
ollama list
```

### 4) Run the demo

From the project root, using your Poetry environment:

```bash
poetry run python demo.py
```

### 5) Expected result

The script will:

1. load the document from [test.txt](test.txt)
2. split it into chunks
3. summarize each chunk with Ollama
4. insert summary embeddings into PGVector
5. query the vector store
6. retrieve and print parent document chunks from the docstore

### 6) Output
Question: What lesson does Tom learn by the end about adventure and consequences?
Why this question: This favors summary-level understanding of the ending.
2026-09-29 12:49:09,277 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-09-29 12:49:09,324 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-09-29 12:49:09,365 - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
```
Naive (raw chunks) top-2 in 0.05s:
  1. chunk_index=0
     One Saturday afternoon Tom wanted to have an adventure because he didn’t want to think about Injun Joe. He went to Huck and said, “I’m going to look for treasure. Do you want to come with me?” Tom Sawyer loves adventures. He has a lot of...
  2. chunk_index=8
     6 Peter had some medicine. He didn’t like it! look at Tom. She went into school. Tom walked away. He didn’t want to go to school now. He was very sad. Joe Harper was near the school. He was sad, too, because his mother was angry with him...

Summary hits used by multi-vector:
  1. chunk_index=23
     Here's a summary of the document: Tom and Becky go on an adventure to a cave with some men in a small boat. They help the men get home, and Tom returns home late on a Tuesday night. The next day, Tom visits Becky's father and tells him a...
  2. chunk_index=1
     The document appears to be a passage from a book, likely "The Adventures of Tom Sawyer" by Mark Twain. Here's a summary: The narrator describes Tom Sawyer, a young boy living with his aunt after his parents' death. Tom is unhappy at scho...

MultiVectorRetriever (summary->parent) top-2 in 0.04s:
  1. chunk_index=23
     happy. They went to the river and waited there. Some men in a small boat came to them. Tom said, “We want to go home. Can you help us?” The men answered, “Yes. We can take you home.” Tom and Becky went in the boat with the men. They arri...
  2. chunk_index=1
     a newspaper there. His stories were in the name of Mark Twain, and people loved them. Later, Samuel lived in New York. His book The Adventures of Tom Sawyer (1876) is about a young boy in a small town in the 1800s. Huck Finn is his frien...

```