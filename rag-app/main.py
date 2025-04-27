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



########################################################################################################################################################
env_mode = os.getenv("ENVIRONMENT_MODE", "development")
if env_mode == "development":
    load_dotenv("../.env.development")

#no need to load ../.env.production because in prod we are using docker and passing it --env-file our ..env.prod
    

#load all variables
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
print(FAISS_INDEX_PATH)
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
#ngrok
NGROK_AUTH_TOKEN=os.getenv("NGROK_AUTH_TOKEN")
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
 
#####################################################################################################################################################
global  custom_retriever
# Create a custom retriever class
class CustomRetriever(BaseRetriever):

    vectorstore: Any
    rerank_llm: Any
    filter_metadata: Optional[Dict[str, Any]] = None  # ✅ Fixed: Added type annotation

    class Config:
        arbitrary_types_allowed = True

    def update_filter_metadata(self,filter_metadata: Dict[str, Any]):
        self.filter_metadata=filter_metadata


    def _get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        #we have to pass filter_metadata through query and then extract  it as well as the original query
        #filter_metadata
        if query=="":
            initial_docs =list(self.vectorstore.docstore._dict.values())
            filtered_docs=[]
            # Filter by metadata if provided
            if self.filter_metadata:
                filtered_docs = [
                    doc for doc in initial_docs
                    #if doc.metadata.get('document_id') == self.filter_metadata.get('document_id')
                    if all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items())
                ]     
            return filtered_docs

        else:
            initial_docs = self.vectorstore.similarity_search(query, k=50)
            filtered_docs=[]
            # Filter by metadata if provided
            if self.filter_metadata:
                filtered_docs = [
                    doc for doc in initial_docs
                    #if doc.metadata.get('document_id') == self.filter_metadata.get('document_id')
                    if all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items())
                ]  
            if len(filtered_docs)==0:
                initial_docs =list(self.vectorstore.docstore._dict.values())
                # Filter by metadata if provided
                if self.filter_metadata:
                    filtered_docs = [
                        doc for doc in initial_docs
                        #if doc.metadata.get('document_id') == self.filter_metadata.get('document_id')
                        if all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items())
                    ]   
                return   filtered_docs
            return rerank_documents(rerank_llm,query, filtered_docs, top_n=num_docs)




class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")
###############################################################################################################################

class QuizItem(BaseModel):
    id: int = Field(..., description="The unique identifier for the quiz item,it should be of type integer")
    question: str = Field(..., description="The quiz question to be answered.")
    options: List[str] = Field(..., description="The list of answer choices.")
    correct_answers_indexes: List[int] = Field(..., description="Index(es) of the correct answer(s).")

class QuizOutput(BaseModel):
    quizzes: List[QuizItem]= Field(..., description="list of return quizzes ")

def create_models():
  ## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
  embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
  instruct_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")
  rerank_llm= ChatOpenAI(temperature=0, model_name="gpt-4.1-nano", max_tokens=4000)
  embed_dims = len(embedder.embed_query("test"))
  llm = instruct_llm
  return embedder, llm, instruct_llm, rerank_llm, embed_dims


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

def format_chunk(doc):
    return (
        f"Paper: {doc.metadata.get('Title', 'unknown')}"
        f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\nPage Body: {doc.page_content}"
    )


#####################################################################

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')


# Debugging Runnable to print the input
def form_input_dict(input_data):
    print(f"Received input data: {input_data}")
    return {'input' : input_data["input"],'num_questions':input_data["num_questions"],'question_type':input_data["question_type"]}  # Pass the input to the next chain step

embedder, llm, instruct_llm, rerank_llm, embed_dims = create_models()
docstore = load_or_create_docstore(FAISS_INDEX_PATH)
docs = []
#docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
#docs = list(docstore.docstore._dict.values())

# 1. Dynamic instruction generator
def get_instruction(inputs):
    question_type = inputs["question_type"]
    if question_type == "0":
        return "Each question must have exactly one correct answer."
    else:
        return "Each question may have multiple correct answers."

# 2. Wrap in RunnableLambda
instruction_node = RunnableLambda(get_instruction)

parser = PydanticOutputParser(pydantic_object=QuizOutput)
format_instructions = parser.get_format_instructions()
chat_prompt = ChatPromptTemplate.from_template(
  """
  {instruction}

  Generate {num_questions} multiple-choice questions from the following content.

  Content:
  {context}

  {format_instructions}

  don't include in your response any additional text, you response must return
  the list of  multiple-choice questions only ! 

  Make sure to generate a correct json format.
  """
)



form_input_dict_node = RunnableLambda(form_input_dict)
long_reorder = RunnableLambda(LongContextReorder().transform_documents)  ## GIVEN




def rerank_documents(rerank_llm,query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    llm=rerank_llm
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]




def retrieve_with_dynamic_filter(input_dict):
    query = input_dict["input"]
    doc_id = input_dict.get("document_id")
    return custom_retriever._get_relevant_documents(
        query=query,
        filter_metadata={"document_id": doc_id} if doc_id else None
    )


custom_retriever = CustomRetriever(vectorstore=docstore,rerank_llm=rerank_llm)

def create_chains():
  # Create the custom retriever
  context_getter = itemgetter('input') | custom_retriever | long_reorder | docs2str
  retrieval_chain = form_input_dict_node  | RunnableAssign({'context' : context_getter})
  generator_chain =     RunnableMap({
        "instruction": instruction_node,
        "context": lambda x: x["context"],
        "format_instructions": lambda x: format_instructions ,
        "num_questions": lambda x: x["num_questions"]
    })|chat_prompt | llm | RunnableLambda(lambda x:print(x.content) or x.content)
  generator_chain = {"output" : generator_chain } | RunnableLambda(output_puller)  ## GIVEN
  rag_chain = retrieval_chain | generator_chain
  return rag_chain,retrieval_chain,generator_chain

rag_chain,retrieval_chain,generator_chain = create_chains()
##################################################################################################################################################################
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

## PRE-ASSESSMENT: Run as-is and see the basic chain in action

add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ASSESSMENT TODO: Implement these components as appropriate

add_routes(
    app,
    generator_chain,
    path="/generator",
)

add_routes(
    app,
    retrieval_chain,
    path="/retriever",
)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

# Async generator function to stream tokens
async def generate_tokens(user_input: str):
    # Generate tokens using the chain
    response = rag_chain.stream(user_input)
    for token in response:
        yield token  # Yield each token one by one

# Your route for streaming response
async def streaming_route(request):
    body = await request.json()
    user_input = body.get('user_input')
    token_stream = generate_tokens(user_input)

    return StreamingResponse(token_stream, media_type="text/plain")

# Register the streaming route using add_routes()
app.add_route("/rag_stream", streaming_route, methods=["POST"])

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



async def file_saver(file: UploadFile):
    try:
        # Ensure upload directory exists
        UPLOAD_DIR = "uploaded_files"
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
        #process and add document to vector database
        encode_pdf(local_file_path,doc_id,s3_doc_path)
        return doc_id,doc_name,s3_doc_path
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
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "documents": []
            }
        )

async def generate_quizzes(input_data):
  response = rag_chain.invoke(input_data)
  try:
    quizzes=json.loads(response)
  except Exception as e:
    print(str(e))
  return  quizzes

def  update_retriever_filter_metadata(filter_metadata):
    custom_retriever.update_filter_metadata(filter_metadata)

@app.post("/generate_quizz")
async def generate_quizz(
    metadata: str = Form(...)
):
    try:
      # Parse the JSON object
      meta = json.loads(metadata)
      num_questions = meta.get("num_questions")
      question_type = meta.get("question_type")
      keywords = meta.get("keywords")
      selected_doc_id = meta.get("selected_doc_id")
      input_query=""
      if len(keywords):
        keyword_str = ", ".join(keywords)
        input_query += f"return every possible chunk related to the following keywords: {keyword_str}"
      update_retriever_filter_metadata({'document_id':int(selected_doc_id)})
      input_data= {'input' :f"{input_query}",'num_questions':num_questions,'question_type':question_type,'keywords':keywords,'selected_doc_id':selected_doc_id}
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
    if env_mode == "production":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        # Start ngrok tunnel
        public_url = ngrok.connect(8000).public_url
        print(f"Public URL {public_url}")

        # Make asyncio work inside thread context (for Colab/Jupyter)
        nest_asyncio.apply()

        # Run the FastAPI app
        uvicorn.run(app, port=8000)


