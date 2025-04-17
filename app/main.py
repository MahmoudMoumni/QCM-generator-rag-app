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
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from pyngrok import ngrok
import nest_asyncio
import uvicorn
import gradio as gr

from pydantic.v1 import BaseModel, Field

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
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
from dotenv import load_dotenv


# Load from .env file

########################################################################################################################################################
load_dotenv()
#####################################################################################################################################################
# Create a custom retriever class
class CustomRetriever(BaseRetriever):

    vectorstore: Any
    rerank_llm: Any

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(rerank_llm,query, initial_docs, top_n=num_docs)

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")
###############################################################################################################################



def create_models():
  ## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
  embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
  instruct_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")
  rerank_llm= ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
  embed_dims = len(embedder.embed_query("test"))
  llm = instruct_llm | StrOutputParser()
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
    print(f"Received Query: {input_data}")
    return {'input' : input_data}  # Pass the input to the next chain step

embedder, llm, instruct_llm, rerank_llm, embed_dims = create_models()
docstore = default_FAISS()
docs = []
#docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
#docs = list(docstore.docstore._dict.values())

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
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



def create_chains():
  # Create the custom retriever
  custom_retriever = CustomRetriever(vectorstore=docstore,rerank_llm=rerank_llm)
  context_getter = itemgetter('input') | custom_retriever | long_reorder | docs2str
  retrieval_chain = form_input_dict_node  | RunnableAssign({'context' : context_getter})
  generator_chain = chat_prompt | llm  ## TODO
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
    print(user_input)
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

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
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
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    #vectorstore = FAISS.from_documents(cleaned_texts, embedder)

    docstore.add_documents(cleaned_texts)



async def file_saver(file: UploadFile):
    # Ensure upload directory exists
    UPLOAD_DIR = "uploaded_files"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #process and add document to vector database
    encode_pdf(file_path)

    return file_path

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    file_path = await file_saver(file)  # Save file
    print(f"File '{file.filename}' uploaded successfully to {file_path}")
    async def response_generator():
        yield "✅ File uploaded successfully! Thanks."

    return StreamingResponse(response_generator(), media_type="text/plain")


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
