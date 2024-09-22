import requests
from bs4 import BeautifulSoup as bs
import re
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
# from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv



load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") #pass your gemini api key here


def text_from_website(url):
    """
    Function to scrap the website using beautifulsoup4, using the content only from the <p>,<h1> and <h2>, also cleaning the
    text by removing unwanted symbols and '\n'.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = bs(response.content, 'html.parser')

        for unwanted in soup(["script", "style", "header", "footer", "nav", "aside"]):
            unwanted.extract()

        content = []
        for tag in soup.find_all(['p', 'h1', 'h2']):
            content.append(tag.get_text())

        combined_text = ' '.join(content).strip()

        cleaned_text = re.sub(r"(\w)-\n(\w)", r"\1\2", combined_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        return cleaned_text
    except Exception as e:
        return f"Error: {e}"


def chunk_text(text):
  '''splitting the corpus into small chunks because LLM have limited context window.
    Splitting text into chunks ensures each chunk fits within this window for better understanding and processing.
    Here we are using RecursiveCharacterTextSplitter from langchain
    '''
  text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200,
            )
  texts = text_splitter.split_text(text)
  return texts



def embedding(texts):
  '''
    Then perform the vectorization on those chunks and convert into embedding and stored in vectorstore
    here im using huggingfaceebmbedding with the model which i used to perform retrieve data
    and used faiss vectorstore to store the vectors
    FAISS demonstrates exceptional proficiency in handling high-dimensional data with remarkable speed and efficiency.
    '''
  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  vector_store = FAISS.from_texts(texts, embeddings)
  return vector_store


website_text=text_from_website('https://botpenguin.com/')

texts=chunk_text(website_text)

vector_store=embedding(texts)

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)

"""
Retrievers: A retriever is an interface that returns documents given
 an unstructured query. It is more general than a vector store.
 A retriever does not need to be able to store documents, only to
 return (or retrieve) them. Vector stores can be used as the backbone
 of a retriever, but there are other types of retrievers as well.
"""
retriever=vector_store.as_retriever()


"""
prompt for the llm to understand and provide the answer as per the user need"""
system_prompt = (
    """
You are an assistant for question-answering tasks.
answer only with in 3 sentence
Use the provided context only to answer the following question:
<context>
{context}
</context>
Question: {input}
"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

''' using ConversationBufferWindowMemory for memory of the chatbot, by window memory we can provide how many 
previous conversation should consider for generating new answer here i'm using 5.
'''
memory = ConversationBufferWindowMemory( k=5) #using conversational buffer memory for provide conversational memory for bot


while True:
  query = input("Ask a question about the website ('END' to exit): ")
  if query.upper() == "END":  # Check for "END/end"
      break
  response = rag_chain.invoke({"input": "what is BotPenguin"})
  answer=response["answer"]
  memory.save_context({"input": query}, {"output": answer})
  print('---'*10)
  print(answer)