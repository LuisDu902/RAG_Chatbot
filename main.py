from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from chainlit.input_widget import Slider
from chainlit.chat_settings import ChatSettings

from model import Model
from basic_model import BasicModel
from upgraded_model import UpgradedModel

temperature_id = "Temperature"
top_k_id = "Top k"
top_p_id = "Top p"
model_id = "model"
settings_id = "settings"
needs_settings_update_id = "needs_settings_update"
sources_id = "sources"
sources_shown_id = "sources_shown"
show_sources_action_name = "show_sources"


@cl.on_chat_start
async def start():
    settings = await ChatSettings(
        [
            Slider(
                id=temperature_id,
                label="Temperature",
                initial=1,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id=top_k_id,
                label="Top k",
                initial=10,
                min=1,
                max=100,
                step=1,
            ),
            Slider(
                id=top_p_id,
                label="Top p",
                initial=0.9,
                min=0,
                max=1,
                step=0.01,
            ),
        ]
    ).send()

    cl.user_session.set(settings_id, settings)
    cl.user_session.set(needs_settings_update_id, True)
    cl.user_session.set(model_id, UpgradedModel())
    cl.user_session.set(sources_id, [])
    cl.user_session.set(sources_shown_id, [])

    start_message = cl.Message(
        content="""# Welcome to the RAG chatbot that helps you understand MDR
This model is designed to assist you in searching and understanding **medical devices regulation**. You can also adjust the model's settings to customize its behavior.
Check the Readme on the left for more information."""
    )
    await start_message.send()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set(settings_id, settings)


def get_model():
    model: Model = cl.user_session.get(model_id)
    needs_settings_update: bool = cl.user_session.get(needs_settings_update_id)
    settings: ChatSettings = cl.user_session.get(settings_id)

    if needs_settings_update:
        temperature = settings[temperature_id]
        top_k = settings[top_k_id]
        top_p = settings[top_p_id]

        model.set_llm_parameters(temperature, top_k, top_p)
        cl.user_session.set(needs_settings_update_id, False)
    return model


@cl.action_callback(show_sources_action_name)
async def show_sources(action: cl.Action):
    id = int(action.value)
    sources_message: cl.Message = cl.user_session.get(sources_id)[id]
    sources_shown: list[bool] = cl.user_session.get(sources_shown_id)
    shown = sources_shown[id]
    if shown:
        await sources_message.remove()
    else:
        await sources_message.send()

    sources_shown[id] = not shown


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    model = get_model()

    model.chat_history.add_user_message(message.content)
    response = model.invoke(message.content)
    if response == "":
        response = """It seems that the LLM model is not able to generate a response for this question
        It may be because of **API limits**. Please try waiting one minute, or asking it in another way."""

    cl_msg = cl.Message(content=response)

    model.chat_history.add_ai_message(response)

    docs = model.docs
    if docs is not None:
        # Format the sources in italics, showing page numbers and the complete source content

        sources_content = "\n\n".join(
            f"- ***Page {doc.metadata.get('page') + 1}**: \"{doc.page_content}\"*"
            for doc in docs
        )
        sources_message = cl.Message(f"\n\n---\n\n**Sources**\n{sources_content}")

        sources = cl.user_session.get(sources_id)
        sources_shown = cl.user_session.get(sources_shown_id)
        action = cl.Action(
            name=show_sources_action_name,
            value=str(len(sources)),
            label="Toggle Sources",
        )

        sources.append(sources_message)
        sources_shown.append(False)
        cl_msg.actions = [action]

    await cl_msg.send()
