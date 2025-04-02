import chainlit as cl
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.media import Image


@cl.on_chat_start
async def on_chat_start():

    agent = Agent(
        model=Ollama(id="llama3.2-vision"), 
        description="You are a helpful AI-powered investment analyst who can analyze financial charts.",
        instructions=["Analyze the images carefully and give precise answers."],
        add_history_to_messages=True,
        markdown=True,
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):

    images = [Image(filepath=file.path) for file in message.elements if "image" in file.mime]

    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    for chunk in await cl.make_async(agent.run)(message.content, images=images, stream=True):
        await msg.stream_token(chunk.get_content_as_string())
    
    await msg.send()