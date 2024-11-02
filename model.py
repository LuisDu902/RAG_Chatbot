from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
import os

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
        separators=[" ", "\n", "\n\n"],
    )
    pages = loader.load_and_split(text_splitter)
    print("Creating the vector database...")
    vectorstore = Chroma.from_documents(
        documents=pages, embedding=embeddings, persist_directory="db"
    )

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible. Mention in which pages the answer is found.

# {context}

# Question: {question}

# Helpful Answer:"""

system_template = """
You are a Q&A chatbot that helps to answer the user's questions about a given document. Always follow these rules to answer the question:

Use the following pieces of context to answer the questions.
If the question is not related to the context, just say it is not related. You can answer the question based on previous context, given on the chat history.
If you don't know the answer to any of the questions, just say that you don't know, don't try to make up an answer.
Always mention in which pages the information you give are found.

<context>
{context}
</context>
"""

# prompt = PromptTemplate.from_template(template)


class MyRunnablePassthrough(Runnable):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def invoke(self, input_data):
        print("HELLO")
        print(input_data[self.name])
        return input_data[self.name]


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        page_number = doc.metadata["page"] + 1
        content_with_page = f"Page {page_number}:\n{doc.page_content}"
        formatted_docs.append(content_with_page)
    return "\n\n".join(formatted_docs)

class Model:  # A model is created per user session
    def __init__(self):
        self.chat_history = ChatMessageHistory()
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_template,
                ),
                (
                    "human",
                    "{question}"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.docs = None

        question_runnable = RunnableLambda(lambda input: input["question"])
        chat_history_runnable = RunnableLambda(lambda input: input["chat_history"])
        self.rag_chain = (
            {
                "context": question_runnable | retriever | self.save_and_format_docs,
                "question": question_runnable,
                "chat_history": chat_history_runnable,
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def save_and_format_docs(self, docs):
        self.docs = docs
        formatted_docs = format_docs(docs)
        return formatted_docs

    def set_llm_parameters(self, temperature, top_k, top_p):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=temperature, top_k=top_k, top_p=top_p
        )

if __name__ == "__main__":
    bad_query = "What is an LLM?"
    query = "What is the medical devices regulation?"
    query_about_history = "Can you repeat your answer from the previous question?"
    query_indirect_history = "Can you provide more context on the previous question?"
    queries = [query, query, query_about_history, query_indirect_history]
    model = Model()

    for query in queries:
        model.chat_history.add_user_message(query)
        response = model.rag_chain.invoke(
            {"question": query, "chat_history": model.chat_history.messages}
        )
        print(response)
        print()
        model.chat_history.add_ai_message(response)

    # Invalid value to test if the values are actually being set
    temperature = -1
    top_k = 5
    top_p = 0.85

    try:
        model.set_llm_parameters(temperature, top_k, top_p)
        print("Parameter values NOT set correctly")
    except ValueError as ve:
        print(
            "Parameter values set correctly. A validation error for temperature should follow:"
        )
        print(ve)
