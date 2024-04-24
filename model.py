import chromadb
import logging
import sys
import time
import config
from typing import List
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.extractors import TitleExtractor,SummaryExtractor
from llama_index.core.node_parser import SimpleNodeParser,SemanticSplitterNodeParser,SentenceSplitter
import tqdm
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from s3fs import S3FileSystem
import s3fs
from llama_index.core import (
    load_index_from_storage,
    load_indices_from_storage,
    load_graph_from_storage,
)
from dotenv import load_dotenv
import os
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.readers.web import WholeSiteReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

aws_key = os.environ["S3_ACCESS_KEY_ID"]
aws_secret = os.environ["S3_ACCESS_SECRET_KEY"]
aws_training_bucket = os.environ["S3_TRAINING_BUCKET"]
r2_account_id = os.environ["R2_ACCOUNT_ID"]
use_s3 = os.environ["USE_S3"]
VECTOR_STORE_PATH = "./vectors"
INPUT_DIR = "./docs/data_to_process"
COLLECTION_NAME = "iollama"
embedding =""


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


global query_engine
query_engine = None
database = f"mongo://{config.MONGO_USER}:{config.MONGO_PASS}@{config.MONGO_HOST}:{config.MONGO_PORT}"
chroma_client = chromadb.PersistentClient(
    path=config.INDEX_PERSIST_DIRECTORY,
)


def init_llm():
    global embedding
    llm = Ollama(model="llama2", request_timeout=300.0)
    embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embedding
    Settings.anonymized_telemetry = False
    return Settings


def add_to_collection(documents: List[Document], metadatas=[], ids=[]):
    chroma_collection = chroma_client.get_or_create_collection("iollama")
    if len(ids) == 0:
        ids = [document.id_ for document in documents if document]
        metadatas = [document.metadata for document in documents if document]
        embeddings = [document.embedding for document in documents if document]

    chroma_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return chroma_collection


def load_store_from_s3(index):
    s3 = s3fs.S3FileSystem(
        key=aws_key,
        secret=aws_secret,
        endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
        s3_additional_kwargs={"ACL": "public-read"},
    )

    # If you're using 2+ indexes with the same StorageContext,
    # run this to save the index to remote blob storage
    index.set_index_id("vector_index")

    # persist index to s3
    s3_bucket_name = "llama-index/storage_demo"  # {bucket_name}/{index_name}
    index.storage_context.persist(persist_dir=s3_bucket_name, fs=s3)

    # load index from s3
    index_from_s3 = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=s3_bucket_name, fs=s3),
        index_id="vector_index",
    )
    return index_from_s3


def document_pipeline(documents, vector_store: ChromaVectorStore,docstore,embed_model):

   # parser = SimpleNodeParser()
   # nodes = parser.get_nodes_from_documents(documents=documents,show_progress=True)
    pipeline = IngestionPipeline(
        transformations=[
            
           SemanticSplitterNodeParser(embed_model=embed_model,buffer_size=3),
            TitleExtractor(),
            embed_model,           
        ],
        
        docstore=docstore,
        # cache=IngestionCache(
        #     cache=RedisCache(
        #         redis_uri="redis://127.0.0.1:6379", collection="test_cache"
        #     )
        # ),
        vector_store=vector_store,
        documents=documents
    )
    # Ingest directly into a vector db
    nodes = pipeline.run(show_progress=True)
    return nodes,vector_store


def load_documents_from_s3() -> List[Document]:
    s3_fs = S3FileSystem(key=aws_key, secret=aws_secret)
    bucket_name = aws_training_bucket

    reader = SimpleDirectoryReader(
        input_dir=bucket_name,
        fs=s3_fs,
        recursive=True,  # recursively searches all subdirectories
        filename_as_id=True,
    )
    documents = []
    # for docs in reader.iter_data():
    # # <do something with the documents per file>
    #  documents.extend(docs)
    logging.info("Loading Data...")
    start = time.perf_counter_ns()
    documents = reader.load_data(show_progress=True, num_workers=4)

    # [logging.debug(doc.metadata) for doc in documents]
    logging.info(f"Loading Data... Completed {len(documents)} documents")
    return documents


def scrapweb(url, embed_model):
    scraper = WholeSiteReader(
        prefix="https://www.sourcegoodfood.com/", max_depth=10
    )  # Example prefix)

    # Start scraping from a base URL
    documents = scraper.load_data(base_url=url)
    add_to_collection(documents, embed_model)
    updated_index = refresh_collection()
    return updated_index


def load_existing_store():
    # initialize client
    db = chroma_client
    # get collection
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def first_load():
    # load some documents
    documents = load_documents()
    # initialize client, setting path to save data
    db = chroma_client

    # create collection
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,show_progress=True)
    return index


def load_documents(local_path=INPUT_DIR,num_workers=4):
    logging.info ("loading documents")
    start = time.perf_counter_ns()

    documents = ""
    if use_s3 == "True":
        documents = load_documents_from_s3()
    else:
        reader = SimpleDirectoryReader(
            input_dir=local_path, recursive=True, filename_as_id=True
        )
        documents = reader.load_data(show_progress=True,num_workers=num_workers)
    end = time.perf_counter_ns()
    logging.info("process  documents in %d ms", (end - start) * 1e-6)
    logging.info ("loading documents ... complete")
    return documents

def update_existing_store(document:Document):
     index = load_existing_store()
     index.update_ref_doc(document)
    
def refresh_collection():

    chroma_client.heartbeat()  # this should work with or without authentication - it is a public endpoint
    print(f"using chorma {chroma_client.get_version()}")

    # Todo fix memory leak
    documents = load_documents(local_path="./docs/fundamentals/pdf-docs")
    
    index = load_existing_store()

    db = chroma_client

    # create collection
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

 
    _,updated_vector_store = document_pipeline(documents=documents,vector_store=vector_store,docstore=storage_context.docstore,embed_model=embedding)
    # use this to set custom chunk size and splitting
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
    # collection = add_to_collection(documents)

    # update_index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)
    # try:
    #    for doc in tqdm.tqdm(documents):
    #     index.insert(doc)    
    # except Exception as e:
    #     print(e)
    #     update_index: VectorStoreIndex = VectorStoreIndex.from_documents(
    #         documents=documentsToProcess,
    #         storage_context=storage_context,
    #         embed_model=embed_model,
    #         show_progress=True,
    #     )
    vector_store.persist(persist_path=VECTOR_STORE_PATH)
    update_index = VectorStoreIndex.from_vector_store(vector_store=updated_vector_store)
    return update_index


def init_index(embed_model):
    logging.info("initializing index documents...")
    new_index=""
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    if chroma_collection.count() == 0 :
        new_index = first_load()
    else:
        new_index = VectorStoreIndex.from_vector_store(vector_store)
    logging.info("initializing index documents... complete")
    return new_index


def init_query_engine(existing_index: VectorStoreIndex,llm="llama2"):
    global query_engine

    # custome prompt template
    template = (
        "You are an advanced AI expert in food regulations, with access to regulations across India, and United state. "
        "Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry accurately"
        ", or cross references where appropriate:\n\n"
        "Question: {query_str}\n\n"
        "Answer vividly, and ensure your response is accurate.\n\n"
        "You will not give any references other than that in your training set"
    )
    qa_template = PromptTemplate(template)

    # build query engine with custom template
    # text_qa_template specifies custom template
    # similarity_top_k configure the retriever to return the top 3 most similar documents,
    # the default value of similarity_top_k is 2
    query_engine = existing_index.as_chat_engine (
        text_qa_template=qa_template, similarity_top_k=3,
        chat_mode="best",verbose=True
    )

    return query_engine


def chat(input_question, user):
    global query_engine

    try:
        response = query_engine.query(input_question)

        logging.info("got response from llm - %s", response)
        for node in response.source_nodes:
            print("-----")
            text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
            print(f"Text:\t {text_fmt} ...")
            print(f"Metadata:\t {node.node.metadata}")
            print(f"Score:\t {node.score:.3f}")

        references = [node.node.metadata for node in response.source_nodes]
        return response.response, references
    except Exception as e:
        if hasattr(e, "message"):
            print(e.message)


def chat_cmd():
    global query_engine

    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == "exit":
            break
        if input_question.lower() == 'refresh':
            refresh_collection()
            continue
        try:
            response, references = chat(input_question, "test-user")
            logging.info("got response from llm - %s \nReferences:\n\n", response)
            references = [
                logging.info(
                    "%d page_number:%s file_name:%s",
                    idx + 1,
                    node["page_label"],
                    node["file_name"],
                )
                for idx, node in enumerate(references)
            ]
        except Exception as e:
            if hasattr(e, "message"):
                print(e.message)


if __name__ == "__main__":
    init_llm()
    index = init_index(Settings.embed_model)
    init_query_engine(index)
    chat_cmd()
