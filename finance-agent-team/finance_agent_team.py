from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.model.ollama import Ollama


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(id="llama3.1:8b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    show_tool_calls=False,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Ollama(id="llama3.1:8b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    show_tool_calls=False,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Ollama(id="llama3.1:8b"),
    name= "Agent Team: Web Agent + Finance Agent",
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=False,
    markdown=True,
)

app = Playground(agents=[agent_team]).get_app()

if __name__ == "__main__":
    serve_playground_app("finance_agent_team:app", reload=True)