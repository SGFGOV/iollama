import os
import logging
import sys
import time
import shutil

import chromadb
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List

from llama_index.core import KnowledgeGraphIndex, load_index_from_storage,load_indices_from_storage
from llama_index.core import StorageContext
from llama_index.readers.web import WholeSiteReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.graph_stores.nebula import NebulaGraphStore

from nebula3.gclient.net import Session
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from model import load_documents
import config

load_dotenv()


SPACE_NAME = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg
use_s3 = os.environ["USE_S3"]
nebula_user = os.environ["NEBULA_USER"]
nebula_password = os.environ["NEBULA_PASSWORD"]
nebula_host = os.environ["NEBULA_HOST"]
nebula_port = os.environ["NEBULA_PORT"]
INPUT_DIR = "./docs"
TEMP_INPUT_DIR = f"./tmp"
TEMP_PROCESSING_DIR =f"./processing"
COLLECTION_NAME = "iollama"
embedding = None
GRAPH_DIR = "./graph_2"


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


query_engine = None
kg_query_engine = None
database = f"mongo://{config.MONGO_USER}:{config.MONGO_PASS}@{config.MONGO_HOST}:{config.MONGO_PORT}"
chroma_client = chromadb.PersistentClient(
    path=config.INDEX_PERSIST_DIRECTORY,
)


def init_space(session: Session):
    command = "CREATE SPACE IF NOT EXISTS llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);"
    session.execute(command)
    time.sleep(10)
    session.execute("USE llamaindex")
    command_create_tag = "CREATE TAG IF NOT EXISTS entity(name string);"
    session.execute(command_create_tag)
    command_create_edge = "CREATE EDGE IF NOT EXISTS relationship(relationship string)"
    time.sleep(10)
    session.execute(command_create_edge)
    time.sleep(10)
    command_create_tag_entity = (
        "CREATE TAG IF NOT EXISTS INDEX entity_index ON entity(name(256))"
    )
    session.execute(command_create_tag_entity)


def create_graph_connection():
    config = Config()
    config.max_connection_pool_size = 10
    # init connection pool
    connection_pool = ConnectionPool()
    # if the given servers are ok, return true, else return false
    ok = connection_pool.init([(nebula_host, nebula_port)], config)

    # option 1 control the connection release yourself
    # get session from the pool

    session = connection_pool.get_session(nebula_user, nebula_password)
    session.execute(f"USE {SPACE_NAME}")
    # select space
    return ok, session, connection_pool


def close_connection(session: Session, connection_pool: ConnectionPool):
    session.release()
    connection_pool.close()


def init_graph_store():
    # define a config
    status, session, connection_pool = create_graph_connection()
    try:
        result = session.execute("USE llamaindex")
        if not result.is_succeeded():
            init_space(session)
    except:
        logging.error("something went wrong")
        return False
    # show tags
    result = session.execute("SHOW TAGS")
    logging.info(result)
    close_connection(session, connection_pool)
    # release session

    return True


def get_graph_store():

    status, session, connection_pool = create_graph_connection()
    os.environ["NEBULA_ADDRESS"] = f"{nebula_host}:{nebula_port}"

    if not status:
        return None, session, connection_pool
    session.execute(f"USE {SPACE_NAME}")
    storage_context:StorageContext = None
    try:
        # storage_context = StorageContext.from_defaults(persist_dir=GRAPH_DIR,graph_store=)
        graph_store = NebulaGraphStore(
            space_name=SPACE_NAME,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
        )
        os.makedirs(GRAPH_DIR, exist_ok=True)
        try:
            storage_context = StorageContext.from_defaults(
                graph_store=graph_store, persist_dir=GRAPH_DIR
            )
        except Exception as e:
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            logging.info(e)

        return storage_context, session, connection_pool
    except Exception as e:
        logging.error(e)


def generate_nebula_graph_single_document_and_persist(
    documents: Document, storage_context: StorageContext, kg_index: KnowledgeGraphIndex
) -> KnowledgeGraphIndex:

    try:
        kg_index._show_progress=True # pylint: disable=W0212
        kg_index.include_embeddings=True
        kg_index.max_triplets_per_chunk=5
        # include_embeddings=False,
        #index._max_object_length=100,
        
        

    
        kg_index.refresh(documents, show_progress=True)
        storage_context.persist(GRAPH_DIR)
    except Exception as e:
        logging.error("something went wrong, unable to refresh...skipping " )
        # for doc in documents:
        #     try:
        #         kg_index.insert(doc)
        #     except Exception as f:
        #         logging.error("unable to insert, " )
    return kg_index, storage_context


def init_index(documents: List[Document], storage_context: StorageContext, session):
    updated_index = KnowledgeGraphIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        max_triplets_per_chunk=5,
        # include_embeddings=False,
        max_object_length=100,
        space_name=SPACE_NAME,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
        show_progress=True,
        session=session,
        num_workers=4,
    )
    storage_context.persist(GRAPH_DIR)
    return updated_index, storage_context


def delete_temp():
    try:
        shutil.rmtree(TEMP_INPUT_DIR)
        logging.info("cleared")

    except PermissionError:
        logging.info("Permission denied.")


def copy_files_to_temp(document_paths: List[str]):
    destination_dir = TEMP_INPUT_DIR
    backup_dir = TEMP_PROCESSING_DIR
    os.makedirs(destination_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    for document_path in tqdm(document_paths, "copying files"):
        # Specify the source file path
        source = document_path
        file_name = source.split("/")[-1]
        destination = destination_dir + "/" + file_name
        backup_dest = backup_dir + "/" + file_name
        # Specify the destination file path
        # Copy the content of source to destination
        try:
            shutil.copyfile(source, destination)
            shutil.copyfile(source,backup_dest)
            log = f'File copied successfully... {file_name}'
            os.remove(source)
            logging.info(log)
        except shutil.SameFileError:
            logging.error("Source and destination are the same file.")
        except IsADirectoryError:
            logging.error("Destination is a directory.")
        except PermissionError:
            logging.error("Permission denied.")
        except Exception:
            logging.error("Error occurred while copying file.")


def update_document_index(storage_context: StorageContext, session: Session,update_index:KnowledgeGraphIndex):

    if os.path.exists(TEMP_INPUT_DIR):
        pdf_parts = load_documents(local_path=TEMP_INPUT_DIR)
        logging.info(f'There are {len(pdf_parts)} nodes to process')
        if update_index != None:
           # old_index = load_index_from_storage(storage_context)
           # update_index = old_index
            skip = False
        else:
            logging.info("unable to loadindices")
            skip = True
            update_index, _ = init_index(
                pdf_parts, storage_context=storage_context, session=session
            )
        if not skip:
            update_index, _ = generate_nebula_graph_single_document_and_persist(
                documents=pdf_parts, storage_context=storage_context, kg_index=update_index
            )
        # for i in enumerate(documents):
        #     if i[0] == 0 and skip:
        #         continue
        #     try:
        #         update_index,_ = generate_nebula_graph_single_document_and_persist(
        #             document=documents[i[0]], storage_context=storage_context, index=update_index
        #         )
        #     except Exception as e:
        #         logging.error(e)
        os.makedirs(GRAPH_DIR, exist_ok=True)
        storage_context.persist(GRAPH_DIR)
        return update_index


def init_kg_index(group_size=5) -> KnowledgeGraphIndex:
    # Specify the directory path
    dir_path = INPUT_DIR
    document_paths: List[str] = []
    session = None
    try:
     storage_context, session, connection_pool = get_graph_store()
    except Exception:
        logging.info("database isn't initilaized")
    if not session:
        status, session, connection_pool = create_graph_connection()
        storage_context, session, connection_pool = get_graph_store()
    batch: List[str] = []
    # Use os.walk to list all files in the directory and its subdirectories
    try:
        old_index = load_index_from_storage(storage_context)
        update_index = old_index
        skip = False
    except Exception as e:
        logging.error(e)
    for root, dirs, files in os.walk(dir_path):
        for idx, file in tqdm(enumerate(files), "generating paths"):
            # Construct the full file path
            file_path = os.path.join(root, file)
            document_paths.insert(0, file_path)
    for idx, path in tqdm(enumerate(document_paths), "processing files"):
        remainder = (idx + 1) % group_size
        batch.insert(0, path)
        if not remainder:
            copy_files_to_temp(batch)
            update_index = update_document_index(
                storage_context=storage_context, session=session,
                update_index=update_index
            )
            batch.clear()
            delete_temp()
    if len(batch) < group_size:
        copy_files_to_temp(batch)
        update_index = update_document_index(
            storage_context=storage_context, session=session,
            update_index=update_index
        )
        batch.clear()
        delete_temp()
    document_paths.clear()
    close_connection(session, connection_pool)
    return update_index, storage_context


def add_documents_to_kg_index(documents: List[Document]) -> KnowledgeGraphIndex:
    documents = load_documents()
    storage_context, session, pool = get_graph_store()
    try:
        old_index = load_index_from_storage(storage_context)
        index = old_index
        for document in documents:
            index.refresh_ref_docs(document)
            storage_context.persist(GRAPH_DIR)
    except Exception as e:
        logging.error(e)
    return index


def kg_query(llm, storage_context, theQuestion):

    kg_query_engine = KnowledgeGraphQueryEngine(
        storage_context=storage_context,
        llm=llm,
        verbose=True,
    )

    response = kg_query_engine.query(str_or_query_bundle=theQuestion)
    return response


def init_kg_llm():
    init_graph_store()
    global embedding
    llm = Ollama(model="llama2", request_timeout=900.0)
    embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embedding
    Settings.anonymized_telemetry = False
    return Settings


def scrapweb(url, prefix="https://www.sourcegoodfood.com/"):
    scraper = WholeSiteReader(prefix=prefix, max_depth=10)  # Example prefix)

    # Start scraping from a base URL
    documents = scraper.load_data(base_url=url)
    return documents


def chat(input_question, user, index: KnowledgeGraphIndex):
    global query_engine

    try:
        storage_context = get_graph_store()
        engine = index.as_chat_engine(llm=Settings.llm)
        response = engine.chat(message=input_question)
        # response = kg_query(llm=Settings.llm,storage_context=storage_context,theQuestion=input_question)

        logging.info("got response from llm - %s", response)
        for node in response.source_nodes:
            logging.info("-----")
            text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
            logging.info(f"Text:\t {text_fmt} ...")
            logging.info(f"Metadata:\t {node.node.metadata}")
            logging.info(f"Score:\t {node.score:.3f}")

        references = [node.node.metadata for node in response.source_nodes]
        return response.response, references
    except Exception as e:
        if hasattr(e, "message"):
            logging.error(e.message)
        else:
            logging.error("something went wrong during chat")
def chat_cmd(index):
    global query_engine

    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == "exit":
            break
        try:
            response, references = chat(input_question, "test-user", index)
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
                logging.error(e.message)
def get_existing_index() -> KnowledgeGraphIndex:
    # Specify the directory path
    dir_path = INPUT_DIR
    document_paths: List[str] = []
    session = None
    old_index= None
    try:
     storage_context, session, connection_pool = get_graph_store()
    except Exception:
        logging.info("database isn't initilaized")
    if not session:
        status, session, connection_pool = create_graph_connection()
        storage_context, session, connection_pool = get_graph_store()
    batch: List[str] = []
    # Use os.walk to list all files in the directory and its subdirectories
    try:
        old_index = load_index_from_storage(storage_context)
    except Exception as e:
        if hasattr(e,"message"):
            logging.error(e.message)
        else :
            logging.error("unable to retrieve store")
    return old_index
init_kg_llm()

if __name__ == "__main__":
    index,_ = init_kg_index(10)
    chat_cmd(index)
