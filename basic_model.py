from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from model import Model, vectorstore

class BasicModel(Model):
    def __init__(self):
        super().__init__()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
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

    def invoke(self, user_question):
        return self.rag_chain.invoke(
            {
                "context": self.docs,
                "question": user_question,
                "chat_history": self.chat_history.messages,
            }
        )


def __test_basic_model():
    bad_query = "What is an LLM?"
    query = "What is the medical devices regulation?"
    query_about_history = "Can you repeat your answer from the previous question?"
    query_indirect_history = "Can you provide more context on the previous question?"
    queries = [query, query_about_history, query_indirect_history]
    model = BasicModel()

    for query in queries:
        model.chat_history.add_user_message(query)
        response = model.invoke(query)
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
            "Parameter values set correctly. A validation error should follow:"
        )
        print(type(ve))

if __name__ == "__main__":
    __test_basic_model()
