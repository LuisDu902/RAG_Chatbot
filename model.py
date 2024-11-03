from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from abc import ABC, abstractmethod

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", task_type="retrieval_document"
)

if os.path.exists("db"):  # Just load the database if it already exists
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
else:
    print("Loading and splitting the document...")
    loader = PDFPlumberLoader("document.pdf")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    pages = loader.load_and_split(text_splitter)
    print("Creating the vector database...")
    vectorstore = Chroma.from_documents(
        documents=pages, embedding=embeddings, persist_directory="db"
    )

system_template = """
You are a Q&A chatbot that helps to answer the user's questions about a given document. Always follow these rules to answer the question:

Use the following pieces of context to answer the questions. The user may reference previous questions, and you can also use the chat history as context.
If the question is not related to the context, just say that it is not related.
If you don't know the answer to any of the questions, just say that you don't know, don't try to make up an answer.
Always mention in which pages the information you give are found.

<context>
{context}
</context>
"""


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        page_number = doc.metadata["page"] + 1
        content_with_page = f"Page {page_number}:\n{doc.page_content}"
        formatted_docs.append(content_with_page)
    return "\n\n".join(formatted_docs)

class Model(ABC):  # A model is created per user session
    def __init__(self):
        self.chat_history = ChatMessageHistory()
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", "{question}"),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.docs = None

    def save_and_format_docs(self, docs):
        self.docs = docs
        formatted_docs = format_docs(docs)
        return formatted_docs

    def set_llm_parameters(self, temperature, top_k, top_p):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=temperature, top_k=top_k, top_p=top_p
        )

    @abstractmethod
    def invoke(self, user_question):
        pass
