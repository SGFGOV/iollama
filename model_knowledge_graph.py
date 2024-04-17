from typing import List
import chromadb
import logging
import sys
import time
import config
from llama_index.core import KnowledgeGraphIndex,load_index_from_storage
from dotenv import load_dotenv
import os
from llama_index.core import StorageContext
from llama_index.readers.web import WholeSiteReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings)
from llama_index.core import StorageContext
from nebula3.gclient.net import Session  

from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import KnowledgeGraphIndex

from llama_index.core import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore
from model import load_documents
from nebula3.gclient.net  import SessionPool
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

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
COLLECTION_NAME = "iollama"
embedding = None
GRAPH_DIR = "./data/graph"


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


global query_engine
global kg_query_engine
kg_query_engine = None
query_engine = None
database = f"mongo://{config.MONGO_USER}:{config.MONGO_PASS}@{config.MONGO_HOST}:{config.MONGO_PORT}"
chroma_client = chromadb.PersistentClient(
    path=config.INDEX_PERSIST_DIRECTORY,
)


def init_space(session:Session):
    command  = 'CREATE SPACE IF NOT EXISTS llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);'
    session.execute(command)
    time.sleep(10)
    session.execute('USE llamaindex')
    command_create_tag='CREATE TAG IF NOT EXISTS entity(name string);'
    session.execute(command_create_tag)     
    command_create_edge = 'CREATE EDGE IF NOT EXISTS relationship(relationship string)'
    time.sleep(10)
    session.execute(command_create_edge)
    time.sleep(10)
    command_create_tag_entity = 'CREATE TAG IF NOT EXISTS INDEX entity_index ON entity(name(256))'
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
    return ok,session,connection_pool
   
def close_connection(session:Session,connection_pool:ConnectionPool):
    session.release()
    connection_pool.close()
def init_graph_store():
        # define a config
    status,session,connection_pool = create_graph_connection()
    try:
       result =  session.execute('USE llamaindex')
       if not result.is_succeeded():
           init_space(session)
    except:
       print('something went wrong')
       return False
    # show tags
    result = session.execute('SHOW TAGS')
    print(result)
    close_connection(session,connection_pool)
    # release session
    
    return True,session

def get_graph_store():

    status,session,connection_pool = create_graph_connection()
    os.environ["NEBULA_ADDRESS"]=f'{nebula_host}:{nebula_port}'
    
    if not status:
        raise Exception("unable to initialize graph store")
    session.execute(f'USE {SPACE_NAME}')
    storage_context = None
    try: 
       # storage_context = StorageContext.from_defaults(persist_dir=GRAPH_DIR,graph_store=)
        graph_store = NebulaGraphStore(
        space_name=SPACE_NAME,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=GRAPH_DIR)
        return storage_context
    except Exception as E:
        print(E)

def generate_nebula_graph(documents:List[Document],storage_context:StorageContext)->KnowledgeGraphIndex:
 
    os.makedirs(GRAPH_DIR,exist_ok=True)
    
    status,session,connection_pool = create_graph_connection()
    result =  session.execute(f'USE {SPACE_NAME}')
    if not result.is_succeeded():
        return
    try:
        
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,

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
            num_workers=4
        )
        storage_context.persist(GRAPH_DIR)
        #storage_context.docstore.persist(GRAPH_DIR)
    except Exception as e:
        print ("something went wrong",e)
        close_connection(session,connection_pool)

    return kg_index,storage_context

def init_kg_index()->KnowledgeGraphIndex:
    documents = load_documents()
    storage_context = get_graph_store()
    try :
        old_index = load_index_from_storage(storage_context)
        index = old_index
    except:
        index = generate_nebula_graph(documents=documents,storage_context=storage_context)
    return index

def add_documents_to_kg_index(documents:List[Document])->KnowledgeGraphIndex:
    documents = load_documents()
    storage_context = get_graph_store()
    try :
        old_index = load_index_from_storage(storage_context)
        index = old_index
        for document in documents:
            index.update_ref_doc(document)
            storage_context.persist(GRAPH_DIR)
    except:
        index = generate_nebula_graph(documents=documents,storage_context=storage_context)
    return index

def kg_query(llm,storage_context,theQuestion):
    
    kg_query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    llm=llm,
    verbose=True,
    
)
    
    response = kg_query_engine.query(
    str_or_query_bundle=theQuestion
)
    return response


def init_kg_llm():
    init_graph_store()
    global embedding
    llm = Ollama(model="llama2", request_timeout=300.0)
    embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embedding
    Settings.anonymized_telemetry = False
    return Settings


def scrapweb(url, prefix="https://www.sourcegoodfood.com/"):
    scraper = WholeSiteReader(
        prefix=prefix, max_depth=10
    )  # Example prefix)

    # Start scraping from a base URL
    documents = scraper.load_data(base_url=url)
    return documents








def chat(input_question, user,index:KnowledgeGraphIndex):
    global query_engine

    try:
        storage_context = get_graph_store()
        engine = index.as_chat_engine(llm=Settings.llm)
        response = engine.chat(message = input_question)
        #response = kg_query(llm=Settings.llm,storage_context=storage_context,theQuestion=input_question)

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


def chat_cmd(index):
    global query_engine

    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == "exit":
            break
        try:
            response, references = chat(input_question, "test-user",index)
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


init_kg_llm()

if __name__ == "__main__":
    index = init_kg_index()
    chat_cmd(index)

