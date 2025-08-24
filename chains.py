from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompts import (
    chat_prompt, triage_prompt, ingestion_prompt,
    takeout_prompt, transaction_prompt, wellness_prompt,
    home_event_prompt, calendar_event_prompt, synthesis_prompt
)

try:
    llm = ChatOllama(model="llama3:8b")

    # Existing chains
    chat_chain = chat_prompt | llm | StrOutputParser()
    triage_chain = triage_prompt | llm
    ingestion_chain = ingestion_prompt | llm

    # New chains for new features
    takeout_chain = takeout_prompt | llm
    transaction_chain = transaction_prompt | llm
    wellness_chain = wellness_prompt | llm | StrOutputParser() # Expects a single string response
    home_event_chain = home_event_prompt | llm
    calendar_event_chain = calendar_event_prompt | llm
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    print("✅ LLM and chains are initialized successfully.")

except Exception as e:
    print(f"❌ FATAL ERROR during chain initialization: {e}")
    llm = None
    chat_chain = None
    triage_chain = None
    ingestion_chain = None
    takeout_chain = None
    transaction_chain = None
    wellness_chain = None
    home_event_chain = None
    calendar_event_chain = None
    synthesis_chain = None
