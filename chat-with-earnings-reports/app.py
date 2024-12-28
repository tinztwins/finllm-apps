import chainlit as cl
from embedchain import Pipeline as App

@cl.on_chat_start
async def on_chat_start():
    app = App.from_config(config_path="config.yml")

    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload an Earnings Report (PDF file) to get started!", accept=["application/pdf"], max_size_mb="10", max_files=1
        ).send()

    text_file = files[0]

    app.add(text_file.path, data_type='pdf_file')
    cl.user_session.set("app", app)

    elements = [
      cl.Pdf(name="pdf", display="inline", path=text_file.path, page=1)
    ]

    await cl.Message(content="Your PDF file:", elements=elements).send()

    await cl.Message(
        content="✅ Successfully added to the knowledge database!"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")
    msg = cl.Message(content="")
    for chunk in await cl.make_async(app.chat)(message.content):
        await msg.stream_token(chunk)
    
    await msg.send()