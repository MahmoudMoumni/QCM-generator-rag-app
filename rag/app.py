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
from langchain.schema.runnable import RunnableLambda, RunnableMap,RunnableBranch
# Output parser lambda
from langchain_core.output_parsers import PydanticOutputParser
from fastapi import FastAPI, File, UploadFile, Form
import json
from fastapi.responses import JSONResponse

from langchain.memory import ConversationBufferMemory
from langchain.schema import   HumanMessage ,AIMessage
########################################################################################################################################################
env_mode = os.getenv("ENVIRONMENT_MODE", "development")
if env_mode == "development":
    load_dotenv("./.env.development")

#load all variables
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
#####################################################################################################################################################
global  quizz_custom_retriever
global  qa_custom_retriever
# Create a custom retriever class
class QuizzCustomRetriever(BaseRetriever):

    vectorstore: Any
    rerank_llm: Any
    filter_metadata: Optional[Dict[str, Any]] = None  # ✅ Fixed: Added type annotation

    class Config:
        arbitrary_types_allowed = True

    def update_filter_metadata(self,filter_metadata: Dict[str, Any]):
        self.filter_metadata=filter_metadata


    def _get_relevant_documents(self, query: str, num_docs=10) -> List[Document]:
        #we have to pass filter_metadata through query and then extract  it as well as the original query
        #filter_metadata
        if query=="":
            initial_docs =list(self.vectorstore.docstore._dict.values())
            print("initiaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaal")
            print(initial_docs)
            filtered_docs=[]
            # Filter by metadata if provided
            if self.filter_metadata:
                filtered_docs = [
                    doc for doc in initial_docs
                    #if doc.metadata.get('document_id') == self.filter_metadata.get('document_id')
                    if all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items())
                ]  
                self.filter_metadata=None   
            print("heloooooooooooooooooooooooooooooooo")
            print(filtered_docs)
            return filtered_docs

        else:
            initial_docs = self.vectorstore.similarity_search(query, k=20)
            filtered_docs=initial_docs
            # Filter by metadata if provided
            if self.filter_metadata:
                filtered_docs = [
                    doc for doc in initial_docs
                    #if doc.metadata.get('document_id') == self.filter_metadata.get('document_id')
                    if all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items())
                ]  
                self.filter_metadata=None   
            if len(filtered_docs)==0:
                initial_docs =list(self.vectorstore.docstore._dict.values())
                # Filter by metadata if provided
                if self.filter_metadata:
                    filtered_docs = [
                        doc for doc in initial_docs
                        #if doc.metadata.get('document_id') == self.filter_metadata.get('document_id')
                        if all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items())
                    ]   
                    self.filter_metadata=None   
                return   filtered_docs
            return rerank_documents(rerank_llm,query, filtered_docs, top_n=num_docs)


class QACustomRetriever(BaseRetriever):

    vectorstore: Any
    rerank_llm: Any

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs=10) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(rerank_llm,query, initial_docs, top_n=num_docs)


class SearchCustomRetriever(BaseRetriever):

    vectorstore: Any
    rerank_llm: Any

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs=7) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=15)
        return rerank_documents_for_search(rerank_llm,query, initial_docs, top_n=num_docs)

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
  rerank_llm= ChatOpenAI(temperature=0, model_name="gpt-4.1-mini", max_tokens=4000)
  embed_dims = len(embedder.embed_query("test"))
  llm = rerank_llm
  return embedder, llm, rerank_llm, embed_dims


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
        doc_id = getattr(doc, 'metadata', {}).get('document_id',"unknown doc")
        doc_page = getattr(doc, 'metadata', {}).get('page')
        #print(f"doc ID {doc_id}")
        #print(f"doc page {doc_page}")
        if doc_id: out_str += f"[Quote from Document with ID {doc_id}] "
        if doc_page: out_str += f"[Data found in page {doc_page}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

def docs2search_results(docs):
    search_results=[]
    for doc in docs:
        search_result={}
        doc_id = getattr(doc, 'metadata', {}).get('document_id',"unknown doc")
        page = getattr(doc, 'metadata', {}).get('page')
        content= getattr(doc, 'page_content', str(doc))
        search_result["doc_id"]=doc_id
        search_result["page"]=page
        search_result["content"]=content
        search_results.append(search_result)
        
    return search_results

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')


# RDebugging Runnable to print the input
def form_input_dict(input_data):
    #print(f"Received input data: {input_data}")
    if input_data["task"]=="generate_quiz":
        return {'input' : input_data["input"],'num_questions':input_data["num_questions"],\
        'question_type':input_data["question_type"],"task":input_data["task"]}  # Pass the input to the next chain step
    else:
        return {'input' : input_data["input"],"task":input_data["task"]}  # Pass the input to the next chain step

embedder, llm, rerank_llm, embed_dims = create_models()
qa_memory = ConversationBufferMemory(return_messages=True)
docstore = load_or_create_docstore(FAISS_INDEX_PATH)
docs = []
form_input_dict_node = RunnableLambda(form_input_dict)
long_reorder = RunnableLambda(LongContextReorder().transform_documents)  ## GIVEN

def rerank_documents(rerank_llm,query: str, docs: List[Document], top_n: int = 6) -> List[Document]:
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

def rerank_documents_for_search(rerank_llm,query: str, docs: List[Document], top_n: int = 15) -> List[Document]:
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
        if score > 0.6:
            scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]

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


quizz_custom_retriever = QuizzCustomRetriever(vectorstore=docstore,rerank_llm=rerank_llm)
qa_custom_retriever = QACustomRetriever(vectorstore=docstore,rerank_llm=rerank_llm)
search_custom_retriever = SearchCustomRetriever(vectorstore=docstore,rerank_llm=rerank_llm)

# Modify your generator chain to store conversation history in memory
def generator_with_memory(inputs):#this function receives list of messages because its after the prompt
    # Generate response with LLM
    response = llm.invoke(inputs)
    # Store LLM response in memory
    qa_memory.chat_memory.add_ai_message("answer :"+str(response.content))

    #print("Memory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n:", memory.load_memory_variables({}))
    return response

def get_chat_history():#we can add limit to take only last few messages
    chat_history =[msg for msg in qa_memory.chat_memory.messages if isinstance(msg, (HumanMessage, AIMessage))]
    return chat_history

# Custom function to combine memory and query
def enhance_query_with_memory(query):
    memory_context = "\n\n".join([msg.content for msg in qa_memory.chat_memory.messages if isinstance(msg, (HumanMessage, AIMessage))])
    combined_query = f"{memory_context}\n\n{query}"
    # Store user input in memory
    qa_memory.chat_memory.add_user_message("question :"+str(query))
    return combined_query

def update_memory_with_user_input(query):

    qa_memory.chat_memory.add_user_message("question :"+str(query))

def create_quizz_rag_chain():
    # Define the chat prompt using System and Human messages
    quizz_chat_prompt = ChatPromptTemplate.from_messages([
        ("system",         """
            You are an assistant that returns ONLY valid JSON object . Do not include any explanation, markdown, or additional \
            text before or after the JSON. The JSON must be valid and parsable.

            Generate {num_questions} multiple-choice questions from the following content. 

            Requirements:
            - Output a **pure JSON list** of objects.
            -  {format_instructions}
            - Do **not** use quotes around `id`.
            - Do **not** add any extra text or explanation before or after the JSON.
            - Do not cut off or leave any question object incomplete.
            - Do **not** wrap the entire output in quotes or markdown.
            - You must return a valid JSON object only with one attribute "quizzes". Anything else will cause an error.

            Content:
            {context}
            """)
        ])

    RPrint =RunnableLambda(lambda x: print(x) or x)
    RDebug=RunnableLambda(lambda x: print("Debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") or x)


    # Create the custom retriever
    context_getter = itemgetter('input') | quizz_custom_retriever | long_reorder |  docs2str
    retrieval_chain = form_input_dict_node | RunnableAssign({'context' : context_getter})|RDebug|RPrint
    

    quizz_config_node=RunnableMap({
        "task" :lambda x: x["task"],
        "instruction": instruction_node,
        "context": lambda x: x["context"],
        "format_instructions": lambda x: format_instructions ,
        "num_questions": lambda x: x["num_questions"]
    })

    # Build the chain dynamically
    generator_chain = (
        quizz_config_node
        | quizz_chat_prompt  # dynamic prompt choice
        | llm
        | RunnableLambda(lambda x: x.content)
    )
    generator_chain = {"output" : generator_chain } | RunnableLambda(output_puller)  ## GIVEN
    quizz_rag_chain =  retrieval_chain | generator_chain 

    return quizz_rag_chain

def create_qa_rag_chain():
    qa_chat_prompt= ChatPromptTemplate.from_messages([
        ("system",  """
        You are a helpful and knowledgeable assistant. Answer the user’s question  based on the information provided in the context.

        Instructions:
        - If the answer **can be found** in the context, provide a **clear and concise** response.
        - If the answer is **not in the context**, respond with: "I’m sorry, I couldn’t find the answer in the provided document."
        - **Do not make up** any information or assumptions outside the context.
        - Use complete sentences in your response.
        - Be factual, direct, and avoid speculation.

        Context:
        {context}
        
        """
         ),
         ("human","""question:{question}""")
    ])

    contextualize_system_prompt = (
    """Given a chat history and the latest user input \
    which might reference context in the chat history, formulate a standalone  question which can be understood \
    without the chat history. Do NOT answer the question, just reformulate it and include the related data from chat history that \
    might be related to the user's last question  otherwise return it as is. Make sure that the formulated question is adressed to the system \
    not to the user,in other words dont complicate question ,your object is to formulate the question and give it context so we can retrieve \
    more related chunks of data from stored documents """
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    RPrint =RunnableLambda(lambda x: print(x) or x)
    RDebug=RunnableLambda(lambda x: print("Debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") or x)

    contextualize_chain=form_input_dict_node | RunnableAssign({'chat_history':lambda x: get_chat_history()}) | \
    RunnableAssign({"contextualized_input": lambda x : ((contextualize_prompt | llm).invoke(x)).content }) |  \
    RunnableAssign({"input":lambda x:x["contextualized_input"]}) | RunnableLambda(lambda x:  update_memory_with_user_input(x["input"]) or x) 


    # Create the custom retriever
    #context_getter = itemgetter('input') | qa_custom_retriever | long_reorder |  docs2str
    context_getter = itemgetter('input') | qa_custom_retriever  |  docs2str
    #retrieval_chain = form_input_dict_node | RunnableAssign({'input': lambda x: enhance_query_with_memory(x["input"])}) | RunnableAssign({'context' : context_getter}) 
    retrieval_chain = form_input_dict_node | RunnableAssign({'context' : context_getter}) 
    

    qa_config_node=RunnableMap({
        "task" :lambda x: x["task"],
        "context": lambda x: x["context"],
        "question":lambda x: x["input"]# we retrieve context with enhanced query and generate answer with enhanced query also
    })

    

    # Build the chain dynamically
    generator_chain = (
        qa_config_node
        | qa_chat_prompt  # dynamic prompt choice
        | RunnableLambda(generator_with_memory)
        | RunnableLambda(lambda x: x.content)
    )
    generator_chain = {"output" : generator_chain } | RunnableLambda(output_puller)  ## GIVEN
    qa_rag_chain = contextualize_chain | RDebug | RPrint | retrieval_chain | generator_chain 

    return qa_rag_chain

def create_search_rag_chain():
    RPrint =RunnableLambda(lambda x: print(x) or x)
    RDebug=RunnableLambda(lambda x: print("Debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") or x)
    context_getter = itemgetter('input') | search_custom_retriever  |  docs2search_results
    retrieval_chain = form_input_dict_node | RunnableAssign({'search_results' : context_getter})
    search_rag_chain = retrieval_chain 
    return search_rag_chain

def  update_retriever_filter_metadata(custom_retriever,filter_metadata):
    custom_retriever.update_filter_metadata(filter_metadata)

##################################################################################################################################################################
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

qa_rag_chain= create_qa_rag_chain()
quizz_rag_chain=create_quizz_rag_chain()
search_rag_chain=create_search_rag_chain()

async def generate_answer(input_data):
    response = qa_rag_chain.invoke(input_data)
    return  response    

@app.post("/generate_answer")
async def get_answer(
    input_data:  Dict[str, Any] 
):
    try:
        print(input_data)
        input_data["task"]="answer_question"
        # Parse the JSON object
        answer=await generate_answer(input_data)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "answer": answer
            }
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "answer": ""
            }
        )


async def generate_quizzes(input_data):
    try:
        response = quizz_rag_chain.invoke(input_data)
        print("response")
        print(response)
        response=json.loads(response)
        quizzes=response["quizzes"]
    except Exception as e:
        print(str(e))
    return  quizzes


@app.post("/generate_quizz")
async def generate_quizz(
    input_data:  Dict[str, Any] 
):
    try:
        print(input_data)
        input_data["task"]="generate_quiz"
        update_retriever_filter_metadata(quizz_custom_retriever,{'document_id':int(input_data["selected_doc_id"])})
        # Parse the JSON object
        quizzes=await generate_quizzes(input_data)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "quizzes": quizzes
            }
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "quizzes": []
            }
        )


async def search(input_data):
    search_results=[]
    try:
        response = search_rag_chain.invoke(input_data)
        print(response)
        search_results=response["search_results"]
    except Exception as e:
        print(str(e))
    return  search_results



@app.post("/search")
async def search_route(
    input_data:  Dict[str, Any] 
):
    try:
        print(input_data)
        input_data["task"]="search"
        #update_retriever_filter_metadata(search_custom_retriever,{'document_id':int(input_data["selected_doc_id"])})
        search_results=await search(input_data)
        print("search_results")
        print(search_results)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "search_results": search_results
            }
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "search_results": []
            }
        )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)


