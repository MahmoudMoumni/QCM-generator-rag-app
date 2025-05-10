import getpass
import os
import sys
import shutil
import asyncio
import random
import textwrap
import numpy as np

from enum import Enum
from functools import partial
from operator import itemgetter
from typing import List, Dict, Any, Tuple,Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

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
import requests


########################################################################################################################################################
env_mode = os.getenv("ENVIRONMENT_MODE", "development")
if env_mode == "development":
    load_dotenv("./.env.development")

#no need to load ../.env.production because in prod we are using docker and passing it --env-file our ..env.prod
    

UPLOAD_DIR= os.getenv("UPLOAD_DIR") 
EMBEDDER_URL = os.getenv("EMBEDDER_URL") # Or localhost:8000 if running locally
RAG_URL = os.getenv("RAG_URL") # Or localhost:8000 if running locally
# S3 Config
S3_AWS_ACCESS_KEY_ID = os.getenv("S3_AWS_ACCESS_KEY_ID")
S3_AWS_SECRET_ACCESS_KEY = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("REGION")
# RDS Config
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

########################################################################################################################################################

# Connect to MySQL server (not to a specific database yet)
conn = mysql.connector.connect(
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT,
    host=DB_HOST,
    database=DB_NAME# now we use db since we created it in previous cell
)
cursor = conn.cursor()


# Initialize the S3 client
s3_client = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=S3_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=S3_AWS_SECRET_ACCESS_KEY
)
 

##################################################################################################################################################################
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)


async def file_saver(file: UploadFile):
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        local_file_path  = os.path.join(UPLOAD_DIR, file.filename)
        with open(local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        doc_name=file.filename
        s3_key = f"documents/{doc_name}"  # This is the S3 "path"
        # Upload
        s3_client.upload_file(local_file_path, BUCKET_NAME, s3_key)
        s3_doc_path = f"s3://{BUCKET_NAME}/{s3_key}"
        #delete file locally and leave only S3 document
        #... code here
        #insert into db
        document = (doc_name, s3_doc_path)
        cursor.execute('''
            INSERT INTO documents (name, path) VALUES (%s, %s)
        ''', document)
        doc_id = cursor.lastrowid  # Get the auto-generated ID
        conn.commit()
    
        data = {
            "local_file_path": local_file_path,
            "doc_id": doc_id,
            "s3_doc_path": s3_doc_path
        }

        response = requests.post(EMBEDDER_URL+"/embedd_document", json=data)

        print("Response:", response.json())
        if response.status_code == 200:
            return doc_id,doc_name,s3_doc_path
        else:
            print(f"Error uploading file")
            return None,None,None
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None,None,None



@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
      doc_id,doc_name,doc_path = await file_saver(file)  # Save file
      #fetch all documents and return them in a response
      cursor.execute("SELECT * FROM documents")
      all_docs = cursor.fetchall()

      # Get column names from cursor description
      columns = [desc[0] for desc in cursor.description]
      # Convert rows to list of dicts
      results = [dict(zip(columns, row)) for row in all_docs]
      # Convert to JSON string (optional)
      all_docs_json = json.dumps(results, indent=2)
      return JSONResponse(
          status_code=200,
          content={
              "status": "success",
              "message": f"File '{doc_name}' uploaded successfully.",
              "documents": results
          }
      )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "documents": []
            }
        )


@app.get("/documents")
async def get_documents():
    try:
      #fetch all documents and return them in a response
      cursor.execute("SELECT * FROM documents")
      all_docs = cursor.fetchall()

      # Get column names from cursor description
      columns = [desc[0] for desc in cursor.description]
      # Convert rows to list of dicts
      results = [dict(zip(columns, row)) for row in all_docs]
      # Convert to JSON string (optional)
      all_docs_json = json.dumps(results, indent=2)
      return JSONResponse(
          status_code=200,
          content={
              "status": "success",
              "message": f"documents retrieved  successfully.",
              "documents": results
          }
      )

    except Exception as e:
        print("retrieving documents error "+str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "documents": []
            }
        )




# Async generator function to stream tokens
async def generate_answer(user_input: str):
    input_data={}
    input_data['input']=user_input
    response=[]
    try:
        #print(input_data)
        response = requests.post(RAG_URL+"/generate_answer", json=input_data)
        if response.status_code == 200:
            print("succes get answer")
            response=response.json()
            answer=response['answer']
            response= [answer]
        else:
            print("failure get answer")
            response= []
    except Exception as e:
        print(str(e))
        
    for token in response:
        yield token  

# Your route for streaming response
async def streaming_route(request):
    body = await request.json()
    user_input = body.get('user_input')
    token_stream =  generate_answer(user_input)
    return StreamingResponse(token_stream, media_type="text/plain")
# Register the streaming route using add_routes()
app.add_route("/rag_stream", streaming_route, methods=["POST"])



async def generate_quizzes(input_data):
    try:
        #print(input_data)
        response = requests.post(RAG_URL+"/generate_quizz", json=input_data)
        if response.status_code == 200:
            print("succes get quizzes")
            response=response.json()
            quizzes=response['quizzes']
            return quizzes
        else:
            print("failure get quizzes")
            return []
    except Exception as e:
        print(str(e))
        return []

@app.post("/generate_quizz")
async def generate_quizz(
    metadata: str = Form(...)
):
    try:
        # Parse the JSON object
        meta = json.loads(metadata)
        print(meta)
        num_questions = meta.get("num_questions")
        question_type = meta.get("question_type")
        keywords = meta.get("keywords")
        selected_doc_id = meta.get("selected_doc_id")
        input_query=""
        if len(keywords):
            keyword_str = ", ".join(keywords)
            input_query += f"return every possible chunk related to the following keywords: {keyword_str}"
        input_data= {'input' :f"{input_query}",'num_questions':num_questions,'question_type':question_type,'keywords':keywords,'selected_doc_id':selected_doc_id}
        quizzes=[]
        #send query to retriever to get context 
        quizzes=await generate_quizzes(input_data)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "quizzes": quizzes
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "quizzes": []
            }
        )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


