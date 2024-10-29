from dotenv import load_dotenv
load_dotenv()

import chainlit as cl   # pip install chainlit
from chainlit.input_widget import Select, Switch, Slider
from chainlit.chat_settings import ChatSettings

from model import Model

temperature_id = "Temperature"
top_k_id = "Top k"
top_p_id = "Top p"
model_id = "model"
settings_id = "settings"
needs_settings_update_id = "needs_settings_update"

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
            # Switch(id="Retrieval", label="Use Retrieval", initial=True),
            # Switch(id="QA", label="Use QA", initial=True),
        ]
    ).send()

    cl.user_session.set(settings_id, settings)
    cl.user_session.set(needs_settings_update_id, True)
    cl.user_session.set(model_id, Model())


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
    msg = cl.Message(content="")

    model = get_model()

    async for chunk in model.rag_chain.astream(message.content):
        await msg.stream_token(chunk)

    await msg.send()

