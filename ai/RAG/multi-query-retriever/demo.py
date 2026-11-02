import hashlib
import json
import logging
import time
from pathlib import Path

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SOURCE_TEXT_PATH = Path(__file__).resolve().parent / "test.txt"
CACHE_DIR = Path(__file__).resolve().parent / ".summary_cache"


def build_excerpt(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def get_summary_cache_path(source_path: str | Path) -> Path:
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    return CACHE_DIR / f"{source_hash}.json"


def load_or_generate_summaries(chunks, source_path: str | Path, summarize_chain):
    cache_path = get_summary_cache_path(source_path)

    if cache_path.exists():
        logger.info("Loading cached summaries from %s", cache_path)
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating summaries for %s", source_path)
    summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})

    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    return summaries


def print_results(title: str, docs: list[Document]) -> None:
    print(f"\n{title}")
    if not docs:
        print("No documents returned.")
        return

    for idx, doc in enumerate(docs, start=1):
        chunk_index = doc.metadata.get("chunk_index", "?")
        print(f"  {idx}. chunk_index={chunk_index}")
        print(f"     {build_excerpt(doc.page_content)}")


def fetch_parent_docs_via_summary(
    summary_vectorstore: PGVector,
    store: InMemoryStore,
    query: str,
    id_key: str,
    k: int = 2,
) -> tuple[list[Document], list[Document]]:
    summary_hits = summary_vectorstore.similarity_search(query, k=k)
    parent_ids = [
        doc.metadata.get(id_key) for doc in summary_hits if doc.metadata.get(id_key)
    ]
    parent_docs = [doc for doc in store.mget(parent_ids) if doc is not None]
    return summary_hits, parent_docs


def main():
    connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
    summary_collection_name = "mv_summaries"
    naive_collection_name = "naive_raw_chunks"
    # Use Ollama locally
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )
    llm_model = OllamaLLM(
        model="llama3.2:1b",
        base_url="http://localhost:11434",
        # num_predict=250,
        temperature=0.2,
    )
    # Load the document
    loader = TextLoader(str(SOURCE_TEXT_PATH), encoding="utf-8")
    docs = loader.load()

    print("length of loaded docs: ", len(docs[0].page_content))
    # Split the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    prompt_text = "Summarize the following document:\n\n{doc}"

    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = (
        {"doc": lambda x: x.page_content} | prompt | llm_model | StrOutputParser()
    )

    summary_cache_path = get_summary_cache_path(SOURCE_TEXT_PATH)
    cache_exists = summary_cache_path.exists()
    start_time = time.time()
    summaries = load_or_generate_summaries(chunks, SOURCE_TEXT_PATH, summarize_chain)
    llm_time = time.time() - start_time

    if cache_exists:
        logger.info(
            "Using cached summaries for %s (loaded in %.2f seconds)",
            SOURCE_TEXT_PATH,
            llm_time,
        )
    else:
        logger.info(f"LLM summarization completed in {llm_time:.2f} seconds")

    # Baseline vector store with direct raw chunk indexing.
    naive_vectorstore = PGVector(
        embeddings=embeddings_model,
        collection_name=naive_collection_name,
        connection=connection,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    # Summary vector store used by the multi-vector retriever.
    summary_vectorstore = PGVector(
        embeddings=embeddings_model,
        collection_name=summary_collection_name,
        connection=connection,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # indexing the summaries in our vector store, whilst retaining the original documents in our document store:
    retriever = MultiVectorRetriever(
        vectorstore=summary_vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 2},
    )

    # Changed from summaries to chunks since we need same length as docs
    doc_ids = [
        hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in chunks
    ]

    # Each summary is linked to the original document by the doc_id
    summary_docs = [
        Document(
            page_content=s,
            metadata={
                id_key: doc_ids[i],
                "chunk_index": chunks[i].metadata["chunk_index"],
            },
        )
        for i, s in enumerate(summaries)
    ]

    logger.info("Adding raw chunks to naive vector store...")
    start_time = time.time()
    naive_vectorstore.add_documents(chunks)
    naive_add_time = time.time() - start_time
    logger.info("Naive vector store add completed in %.2f seconds", naive_add_time)

    # Add summary documents to the vector store for similarity search.
    # Parent mapping uses metadata[id_key], so explicit vector IDs are not required.
    logger.info("Adding summaries to vector store...")
    start_time = time.time()
    retriever.vectorstore.add_documents(summary_docs)
    summary_add_time = time.time() - start_time
    logger.info("Summary vector store add completed in %.2f seconds", summary_add_time)

    # Store the original documents in the document store, linked to their summaries via doc_ids
    # This allows us to first search summaries efficiently, then fetch the full docs when needed
    retriever.docstore.mset(list(zip(doc_ids, chunks)))

    # vector store retrieves the summaries
    queries = [
        # (
        #     "Where does the story shift from playful adventure to real danger?",
        #     "This asks for a thematic transition rather than a single exact sentence.",
        # # )
        # (
        #     "Why does the treasure hunt become more threatening after the haunted house episode?",
        #     "This links cause and mood across connected events.",
        # ),
        # (
        #     "Which location represents freedom for Tom and Huck, and which represents fear?",
        #     "This compares roles of two locations (river vs cave).",
        # ),
        # (
        #     "who is inviting Huck to come to school and play with the boys",
        #     "",
        # ),
        # ,
        (
            "What lesson does Tom learn by the end about adventure and consequences?",
            "This favors summary-level understanding of the ending.",
        ),
    ]

    naive_search_times = []
    retriever_times = []

    for query, rationale in queries:
        print("\n" + "=" * 80)
        print(f"Question: {query}")
        print(f"Why this question: {rationale}")

        start_time = time.time()
        naive_docs = naive_vectorstore.similarity_search(query, k=2)
        naive_time = time.time() - start_time
        naive_search_times.append(naive_time)

        start_time = time.time()
        multi_vector_docs = retriever.invoke(query)
        retriever_time = time.time() - start_time

        summary_hits, fallback_multi_docs = fetch_parent_docs_via_summary(
            summary_vectorstore=summary_vectorstore,
            store=store,
            query=query,
            id_key=id_key,
            k=2,
        )

        if not multi_vector_docs and fallback_multi_docs:
            logger.warning(
                "retriever.invoke returned no parent docs; using summary->docstore fallback for display."
            )
            multi_vector_docs = fallback_multi_docs
        retriever_times.append(retriever_time)

        print_results(
            f"Naive (raw chunks) top-2 in {naive_time:.2f}s:",
            naive_docs,
        )
        print_results("Summary hits used by multi-vector:", summary_hits)
        print_results(
            f"MultiVectorRetriever (summary->parent) top-2 in {retriever_time:.2f}s:",
            multi_vector_docs,
        )

    # Summary of performance metrics
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY (comparison run)")
    logger.info("=" * 60)
    logger.info(f"LLM Summarization Time: {llm_time:.2f}s")
    logger.info(f"Naive Add Time: {naive_add_time:.2f}s")
    logger.info(f"Summary Add Time: {summary_add_time:.2f}s")
    logger.info(
        "Avg Naive Query Time: %.2fs",
        sum(naive_search_times) / len(naive_search_times),
    )
    logger.info(
        "Avg Multi-Vector Query Time: %.2fs",
        sum(retriever_times) / len(retriever_times),
    )
    logger.info("=" * 60)
    logger.info("Note: Ollama tokens per second is available in Ollama logs")


if __name__ == "__main__":
    main()
