import getpass
import os
import sys
import shutil
import asyncio
import random
import textwrap


from enum import Enum
from functools import partial
from operator import itemgetter
from typing import List, Dict, Any, Tuple,Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from pyngrok import ngrok
import nest_asyncio
import uvicorn
import gradio as gr

from pydantic.v1 import BaseModel, Field

#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough,
    RunnableAssign
)
from langchain_core.retrievers import BaseRetriever

from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
    NVIDIAEmbeddings,
    NVIDIARerank
)
from langserve import add_routes
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

from pydantic import BaseModel, Field,RootModel
from typing import List
import mysql.connector
from langchain.schema.runnable import RunnableLambda, RunnableMap
# Output parser lambda
from langchain_core.output_parsers import PydanticOutputParser
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import json
import time
from fastapi.responses import JSONResponse
import boto3

from langchain.vectorstores import Chroma

########################################################################################################################################################
env_mode = os.getenv("ENVIRONMENT_MODE", "development")
if env_mode == "development":
    load_dotenv("./.env.development")

#no need to load ../.env.production because in prod we are using docker and passing it --env-file our ..env.prod
    

#load all variables
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
print(FAISS_INDEX_PATH)


########################################################################################################################################################

def get_embeddings_params():
  embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
  embed_dims = len(embedder.embed_query("test"))
  return embedder,  embed_dims


def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def load_or_create_docstore(faiss_index_path):
    if os.path.exists(f"{faiss_index_path}/index.faiss") and os.path.exists(f"{faiss_index_path}/index.pkl"):
        return FAISS.load_local(faiss_index_path, embeddings=embedder,allow_dangerous_deserialization=True)
    return default_FAISS()

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore


embedder, embed_dims = get_embeddings_params()
docstore = load_or_create_docstore(FAISS_INDEX_PATH)


##################################################################################################################################################################
app = FastAPI(
  title="retriever Server",
  version="1.0",
  description="retriever server apis",
)



def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def encode_pdf(path,document_id,s3_doc_path,chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    # Add document_id to each chunk's metadata
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source_file"] = s3_doc_path
    cleaned_texts = replace_t_with_space(chunks)

    # Create embeddings and vector store
    #vectorstore = FAISS.from_documents(cleaned_texts, embedder)

    docstore.add_documents(cleaned_texts)
    # ✅ Save to local directory
    docstore.save_local(FAISS_INDEX_PATH)    






class DocumentMetadata(BaseModel):
    local_file_path: str
    doc_id: int
    s3_doc_path: str

@app.post("/embedd_document")
async def embedd_document(
    metadata: DocumentMetadata
):
    try:
      print(f"Received: {metadata}")
      doc_id=metadata.doc_id
      local_file_path=metadata.local_file_path
      s3_doc_path=metadata.s3_doc_path
      encode_pdf(local_file_path,doc_id,s3_doc_path)
      return JSONResponse(
            status_code=200,
            content={
                "status": "success"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error"
            }
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


