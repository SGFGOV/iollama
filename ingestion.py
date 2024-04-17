from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.ingestion.cache import RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.readers.google import GoogleDriveReader

loader = GoogleDriveReader()

vector_store = RedisVectorStore(
    index_name="redis_vector_store",
    index_prefix="vectore_store",
    redis_url="redis://localhost:6379",
)

def load_data(folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs


docs = load_data(folder_id="1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5")
# print(docs)

cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)

# Optional: clear vector store if exists
if vector_store._index_exists():
    vector_store.delete_index()

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)



index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)