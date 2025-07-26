# Converted from 01a.Summarize Private Documents Using RAG.ipynb
# All markdown cells are included as comments for context and learning reference.
# You can run this script as a regular Python file after installing the required libraries.

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>

# # **Summarize Private Documents Using RAG, LangChain, and LLMs**

# ##### Estimated time needed: **45** minutes

# ... (rest of markdown as comments)

# ## Setup

# For this lab, you are going to use the following libraries:
# * ibm-watsonx-ai
# * langchain
# * langchain-ibm
# * huggingface
# * huggingface-hub
# * sentence-transformers
# * chromadb
# * wget

# ---
# Installing required libraries
# You can uncomment the following lines to install the required libraries if not already installed.

# import sys
# !{sys.executable} -m pip install --user "ibm-watsonx-ai==0.2.6"
# !{sys.executable} -m pip install --user "langchain==0.1.16"
# !{sys.executable} -m pip install --user "langchain-ibm==0.1.4"
# !{sys.executable} -m pip install --user "huggingface==0.0.1"
# !{sys.executable} -m pip install --user "huggingface-hub==0.23.4"
# !{sys.executable} -m pip install --user "sentence-transformers==2.5.1"
# !{sys.executable} -m pip install --user "chromadb"
# !{sys.executable} -m pip install --user "wget==3.2"

# ---
# Importing required libraries

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# The following imports are for IBM watsonx.ai cloud models, which you may want to replace with local models for full offline use.
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget

# ---
# ## Preprocessing
# ### Load the document

filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# Use wget to download the file
wget.download(url, out=filename)
print('file downloaded')

# View the document
with open(filename, 'r') as file:
    contents = file.read()
    print(contents)

# ---
# ### Splitting the document into chunks
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))

# ---
# ### Embedding and storing
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested')

# ---
# ## LLM model construction
# NOTE: The following code uses IBM watsonx.ai cloud models. For local/offline use, you should replace this with a local LLM (e.g., HuggingFace Transformers, llama.cpp, etc.)

model_id = 'google/flan-ul2'
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "api_key": "your api key here"  # Uncomment and set if running locally with your own IBM Cloud account
}
project_id = "skills-network"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

flan_ul2_llm = WatsonxLLM(model=model)

# ---
# ## Integrating LangChain
qa = RetrievalQA.from_chain_type(
    llm=flan_ul2_llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(), 
    return_source_documents=False
)
query = "what is mobile policy?"
print(qa.invoke(query))

# ---
# Try a high-level question
qa = RetrievalQA.from_chain_type(
    llm=flan_ul2_llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(), 
    return_source_documents=False
)
query = "Can you summarize the document for me?"
print(qa.invoke(query))

# ---
# Try another model (LLAMA_3_70B_INSTRUCT)
model_id = 'meta-llama/llama-3-3-70b-instruct'
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}
project_id = "skills-network"
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
llama_3_llm = WatsonxLLM(model=model)

qa = RetrievalQA.from_chain_type(
    llm=llama_3_llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(), 
    return_source_documents=False
)
query = "Can you summarize the document for me?"
print(qa.invoke(query))

# ---
# Prompt template for better control
prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.\n\n{context}\n\nQuestion: {question}\n"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=llama_3_llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(), 
    chain_type_kwargs=chain_type_kwargs, 
    return_source_documents=False
)
query = "Can I eat in company vehicles?"
print(qa.invoke(query))

# ---
# Conversation with memory
memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
qa = ConversationalRetrievalChain.from_llm(
    llm=llama_3_llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(), 
    memory = memory, 
    get_chat_history=lambda h : h, 
    return_source_documents=False
)
history = []
query = "What is mobile policy?"
result = qa.invoke({"question":query}, {"chat_history": history})
print(result["answer"])
history.append((query, result["answer"]))

query = "List points in it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])
history.append((query, result["answer"]))

query = "What is the aim of it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])

# ---
# Agent function for interactive Q&A

def qa_agent():
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llama_3_llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        memory = memory, 
        get_chat_history=lambda h : h, 
        return_source_documents=False
    )
    history = []
    while True:
        query = input("Question: ")
        if query.lower() in ["quit","exit","bye"]:
            print("Answer: Goodbye!")
            break
        result = qa({"question": query}, {"chat_history": history})
        history.append((query, result["answer"]))
        print("Answer: ", result["answer"])

# Uncomment to run the agent interactively
# qa_agent()

# ---
# Exercises are included as comments for further practice.
# See the original notebook for more details and solutions. 