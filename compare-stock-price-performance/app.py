import chainlit as cl
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from typing import Annotated
import matplotlib.pyplot as plt
import matplotlib
import yfinance

matplotlib.use("Agg")  # Non-interactive backend: headless mode

# Configure the API endpoint for your model
config_list = [
    {
        "model": "llama3.1:latest",
        "api_type": "ollama",
    }
]


def get_stock_prices(stock_symbols, 
                     start_date, 
                     end_date):
    stock_data = yfinance.download(
        stock_symbols, 
        start=start_date, 
        end=end_date
    )
    return stock_data.get("Close")

def plot_ytd_gains(stock_symbols: Annotated[list[str], "The stock symbols to get the prices for. (list[str])"], 
                   start_date: Annotated[str, "Format: YYYY-MM-DD"], 
                   end_date: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    """Plot the stock gain YTD for the given stock symbols.

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format 
        'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
        str: A status message
    """
    stock_data_list = list()
    for i in stock_symbols:
        stock_price_data = get_stock_prices(i, start_date, end_date)
        stock_ytd_gain = (stock_price_data - stock_price_data.iloc[0]) / stock_price_data.iloc[0] * 100
        stock_data_list.append(stock_ytd_gain)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for df in stock_data_list:
        plt.plot(df, label=df.columns.values[0])
    plt.title('Stock Price Performance YTD')
    plt.xlabel('Date')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.ylabel('Gain in %')
    plt.legend()
    plt.grid(True)
    cl.user_session.set("figure", fig)

    return "The task is done."


@cl.on_chat_start
async def on_chat_start():
    # Define the financial analyst agent that suggests tool calls.
    financial_analyst = ConversableAgent(
        name="Financial Analyst",
        system_message="You are a helpful AI assistant."
        "You can help with plotting financial charts."
        "Don't plot a chart when no stock symbols are given."
        "Return 'TERMINATE' when the task is done.",
        llm_config={"config_list": config_list},
    )

    # This is the assistant agent.
    assistant_agent = ConversableAgent(
        name="Assistant",
        system_message="You are a helpful AI assistant."
        "You can help with general queries and questions about the stock market."
        "Add 'TERMINATE' to every answer.",
        code_execution_config=False,
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",
    )

    # The user proxy agent can interact with the assistant agent and the financial analyst.
    user_proxy_agent = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
    )

    # Register the function to the two agents.
    register_function(
        plot_ytd_gains,
        caller=financial_analyst, 
        executor=user_proxy_agent,
        name="plot_ytd_gains",
        description="A function to plot the stock gain YTD for the given stock symbols.",  # A description of the tool.
    )

    # Create a GroupChat
    groupchat = GroupChat(agents=[user_proxy_agent, 
                                  assistant_agent, 
                                  financial_analyst], 
                        messages=[], 
                        max_round=5,
                        allowed_or_disallowed_speaker_transitions={
            user_proxy_agent: [assistant_agent, 
                               financial_analyst],
            assistant_agent: [user_proxy_agent],
            financial_analyst: [user_proxy_agent],
        },
        speaker_transitions_type="allowed",)
    manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

    cl.user_session.set("manager", manager)
    cl.user_session.set("user_proxy_agent", user_proxy_agent)


@cl.on_message
async def on_message(message: cl.Message):
    manager = cl.user_session.get("manager")
    user_proxy_agent = cl.user_session.get("user_proxy_agent")

    groupchat_result = await cl.make_async(user_proxy_agent.initiate_chat)(manager, message=message.content)
    fig = cl.user_session.get("figure")

    if fig:
        elements = [
            cl.Pyplot(name="plot", figure=fig, display="inline"),
        ]
        await cl.Message(
            content="You can see the stock gain YTD plot below:",
            elements=elements,
        ).send()
        cl.user_session.set("figure", 0)
    else:
        msg = cl.Message(content="")
        for chunk in groupchat_result.summary:
            await msg.stream_token(chunk)
        await msg.send()