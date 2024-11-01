from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
import os

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")

if os.path.exists("db"):    # Just load the database if it already exists
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
else:
    loader = PDFPlumberLoader("document.pdf")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True, separators=[" ", "\n", "\n\n"]
    )
    pages = loader.load_and_split(text_splitter)
    vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory="db")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible. Mention in which pages the answer is found.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        page_number = doc.metadata["page"] + 1 
        content_with_page = f"Page {page_number}:\n{doc.page_content}"
        formatted_docs.append(content_with_page)
    return "\n\n".join(formatted_docs)

class Model:    # A model is created per user session
    def format_and_save_docs(self, docs):
        self.docs = docs
        formatted_docs = format_docs(docs)
        return formatted_docs

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.docs = None
        self.rag_chain = (
            {"context": retriever | self.format_and_save_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def set_llm_parameters(self, temperature, top_k, top_p):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=temperature, top_k=top_k, top_p=top_p
        )

if __name__ == "__main__":
    query = "Describe the use of harmonised standards"
    model = Model()
    
    result = model.qa_chain.invoke({"query": query})
    print("Result for QA chain:")
    print(result)
    print()

    result = model.rag_chain.invoke("What is an LLM?")
    print("Result for normal chain:")
    print(result)
    print()

    # Invalid value to test if the values are actually being set
    temperature = -1
    top_k = 5
    top_p = 0.85

    try:
        model.set_llm_parameters(temperature, top_k, top_p)
        print("Parameter values NOT set correctly")
    except ValueError as ve:
        print("Parameter values set correctly. A validation error for temperature should follow:")
        print(ve)
