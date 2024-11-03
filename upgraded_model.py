from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model import Model, vectorstore

class UpgradedModel(Model):
    def formatted_chat_history(self):
        result = ""
        question_id = 1
        for message in self.chat_history.messages:
            if message.type == "human":
                result += f"{question_id}. Human: {message.content}\n"
                question_id += 1
            else:
                result += f"AI: {message.content}\n"
        return result

    def __init__(self):
        super().__init__()

        metadata_field_info = [
            AttributeInfo(
                name="page",
                type="int",
                description="The page number of the document",
            )
        ]
        document_content_description = "Medical devices regulation"
        query_with_history_template = """
        You are an AI assistant that helps a user query a document.
        The user makes some questions and you create the queries to find the parts of document that are most relevant to the questions.
        The search will be performed by similarity so you need to provide a query similar to the contents that are in the document.
        You do not have the context of the document. You will be making the queries based on the questions and history to get the context from the document.
        This is the chat history of the conversation until now:
        <history>
        {chat_history}
        </history>

        Take into account the context of the chat history when preparing the query for the following question:
        Prepare a query for this question. Output only the query as a natural language question.
        Question: {question}"""

        retriever_prompt_template = PromptTemplate.from_template(
            query_with_history_template
        )

        self.retriever = SelfQueryRetriever.from_llm(
            self.llm,
            vectorstore,
            document_content_description,
            metadata_field_info,
            verbose=True,
        )

        # Ask the LLM to generate a question based on the chat history and the user question
        # And ask the LLM to perform a search (can filter) based on the question generated
        self.retriever_chain = retriever_prompt_template | self.llm | StrOutputParser()

        # Normal RAG chain with the documents from the retrieval
        self.rag_chain = self.qa_prompt | self.llm | StrOutputParser()

    def __get_llm_db_question(self, user_question):
        return self.retriever_chain.invoke(
            {"question": user_question, "chat_history": self.formatted_chat_history()}
        )

    def invoke(self, user_question):
        llm_db_question = self.__get_llm_db_question(user_question)
        self.docs = self.retriever.invoke(llm_db_question)
        return self.rag_chain.invoke(
            {
                "context": self.docs,
                "question": llm_db_question,
                "chat_history": self.chat_history.messages,
            }
        )


def __test_upgraded_model():
    model = UpgradedModel()

    queries = [
        "What is in the first page?",
        "What is the medical devices regulation?",
        "Can you further explain the first question?",
    ]
    for query in queries:
        model.chat_history.add_user_message(query)
        response = model.invoke(query)
        model.chat_history.add_ai_message(response)

        print(response)
        print("\n\n---\n\n")


if __name__ == "__main__":
    __test_upgraded_model()

