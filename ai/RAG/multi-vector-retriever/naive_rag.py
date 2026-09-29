import logging

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "naive_rag_raw_chunks"

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

# Load source document
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()
logger.info("Loaded %s document(s)", len(docs))

# Split into raw chunks, without summaries
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
logger.info("Split document into %s chunks", len(chunks))

# Store raw chunks directly in a separate collection for naive RAG baseline
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    pre_delete_collection=True,
)

# Add the raw chunk documents directly to the vector field
logger.info("Adding raw chunks to vector store...")
vectorstore.add_documents(chunks)

# Query the naive baseline with a set of fact-vs-topic comparisons
queries = [
    "Did Tom have feelings for Becky Thatcher?",
    "What happens with Tom and Huck near the river?",
    "Who is the dangerous man connected to the haunted house?",
    "What do the boys do after finding the treasure map?",
]

for query in queries:
    logger.info("Running naive RAG query: %s", query)
    retrieved = vectorstore.similarity_search(query, k=3)

    print("\n=== Naive RAG query ===")
    print(query)
    print("\n=== Retrieved chunks ===")
    for i, doc in enumerate(retrieved, start=1):
        print(f"\n--- chunk {i} ---")
        print(doc.page_content)
        print(f"metadata: {doc.metadata}")

    print(f"\nRetrieved chunk count: {len(retrieved)}")
