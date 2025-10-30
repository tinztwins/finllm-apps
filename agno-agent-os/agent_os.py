from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.yfinance import YFinanceTools
from agno.os import AgentOS

investment_agent = Agent(
    id="investment-agent",
    model=Ollama(id="gpt-oss:latest"),
    tools=[
        YFinanceTools(),
    ],
    description="You are an investment analyst that researches stock prices, company infos, stock fundamentals, analyst recommendations and historical prices",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
)

agent_os = AgentOS(
    id="agent-os",
    description="AgentOS",
    agents=[investment_agent],
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agent_os:app", reload=True)