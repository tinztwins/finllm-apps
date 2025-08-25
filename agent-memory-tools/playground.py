from agno.agent import Agent
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.ollama import Ollama
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground

memory = Memory(
    model=Ollama(id="gpt-oss:latest"),
    db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db"),
    delete_memories=True,
    clear_memories=True,
)

agent = Agent(
    model=Ollama(id="gpt-oss:latest"),
    tools=[
        YFinanceTools(stock_price=True,
                      company_info=True, 
                      stock_fundamentals=True, 
                      analyst_recommendations=True,
                      historical_prices=True),
    ],
    user_id="tinztwins",
    description="You are an investment analyst that researches stock prices, company infos, stock fundamentals, analyst recommendations and historical prices",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    memory=memory,
    enable_agentic_memory=True,
    markdown=True
)


playground = Playground(agents=[agent])
app = playground.get_app()

if __name__ == "__main__":
    playground.serve("playground:app", reload=True)





# if __name__ == "__main__":
#     # This will create a memory that "ava's" favorite stocks are NVIDIA and TSLA
#     agent.print_response(
#         "My favorite stocks are NVIDIA and TSLA",
#         stream=True,
#         show_full_reasoning=True,
#         stream_intermediate_steps=True,
#     )
#     # This will use the memory to answer the question
#     agent.print_response(
#         "Can you compare my favorite stocks?",
#         stream=True,
#         show_full_reasoning=True,
#         stream_intermediate_steps=True,
#     )



# My favorite stocks are Palantir (PLTR) and Tesla (TSLA). Tell me the CEOs of both companies.
# Can you tell me the strengths and weaknesses of my favorite stocks?
# Forget my favorite stocks.