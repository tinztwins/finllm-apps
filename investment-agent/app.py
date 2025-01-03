import chainlit as cl
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.ollama import Ollama


@cl.on_chat_start
async def on_chat_start():
    agent = Agent(model=Ollama(id="llama3.1:8b"), 
                  tools=[YFinanceTools(stock_price=True, 
                                       company_info=True, 
                                       stock_fundamentals=True, 
                                       analyst_recommendations=True,
                                       historical_prices=True), 
                        DuckDuckGo()],
                  description="You are an investment analyst that researches stock prices, company infos, stock fundamentals, analyst recommendations and historical prices",
                  instructions=["Format your response using markdown and use tables to display data where possible."],)

    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    for chunk in await cl.make_async(agent.run)(message.content, stream=True):
        await msg.stream_token(chunk.get_content_as_string())
    
    await msg.send()