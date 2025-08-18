from fastapi import FastAPI
from routers import ingestion_agents, context_agents

app = FastAPI(
    title="Local Agentic Core API",
    description="An API for interacting with local AI agents and models."
)

app.include_router(ingestion_agents.router)
app.include_router(context_agents.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Local Agentic Core API"}
