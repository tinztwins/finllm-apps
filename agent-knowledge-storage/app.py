from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.ollama import Ollama
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.playground import Playground

knowledge = WebsiteKnowledgeBase(
    urls=["https://tinztwinshub.com/"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="tinztwinshub_docs",
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="all-minilm:latest", dimensions=384),
    ),
)

storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")

agent = Agent(
    name="Tinz Twins Hub Assist",
    model=Ollama(id="llama3.1:8b"),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_history_to_messages=True,
    markdown=True,
)

playground = Playground(agents=[agent])
app = playground.get_app()

if __name__ == "__main__":
    agent.knowledge.load(recreate=True)
    playground.serve("app:app", reload=True)