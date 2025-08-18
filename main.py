# main.py

import json
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import requests
import langextract


# --- 1. Pydantic Models (Data Contracts) ---
class QueryRequest(BaseModel):
    query: str

class EmailTriageRequest(BaseModel):
    from_sender: str
    subject: str
    body: str

class TriageResponse(BaseModel):
    category: str
    urgency: str
    summary: str
    action_item: Optional[str] = None

class MemorizeRequest(BaseModel):
    text: str

class ProcessUrlRequest(BaseModel):
    url: str

class IngestionResponse(BaseModel):
    summary: str
    tags: list[str]
    original_url: Optional[str] = None


# --- 2. FastAPI App Initialization ---
app = FastAPI(
    title="Local Agentic Core API",
    description="An API for interacting with local AI agents and models."
)


# --- 3. Global Objects (LLM & Chains) ---
try:
    llm = ChatOllama(model="llama3:8b")

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant providing concise, expert-level answers."),
        ("user", "{input}")
    ])
    chat_chain = chat_prompt | llm | StrOutputParser()

    triage_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert executive assistant. Your task is to triage an incoming email and provide a structured response in JSON format.

**GUIDELINES:**
- Analyze the sender, subject, and body to make your decision.
- Use the specific categories provided below. Avoid using "Other" unless absolutely necessary.
- A direct request or question in the email body should be an `action_item`.
- Security alerts are always "High" urgency. Transaction alerts are "Medium".

**AVAILABLE CATEGORIES:**
- "Transaction Alert": Notifications of funds transfers, payments, e-receipts (UOB, DBS, Grab, PayLah!).
- "Billing/Invoice": Bills, e-statements, or invoices that require payment or review (e.g., from utilities, CPF).
- "Security Alert": Notifications about new logins, app connections, or password changes.
- "Personal Appointment": Reminders for personal events, appointments, or bookings (e.g., gym, restaurants).
- "Project Update": Work-related updates, newsletters from professional sites like Medium.
- "Newsletter/Marketing": Promotional emails, sales, marketing content, and general newsletters.
- "Spam": Unsolicited or irrelevant junk mail.
- "Other": Use only as a last resort.

**EXAMPLES:**
- **Input:** From: unialerts@uobgroup.com, Subject: UOB Personal Internet Banking Notification Alerts
  **Output:** {{"category": "Transaction Alert", "urgency": "Medium", "summary": "Notification of a funds transfer from UOB.", "action_item": null}}
- **Input:** From: account-security-noreply@accountprotection.microsoft.com, Subject: New app(s) connected to your Microsoft account
  **Output:** {{"category": "Security Alert", "urgency": "High", "summary": "A new application was connected to the Microsoft account.", "action_item": "Review the connected app and remove it if it was not authorized."}}
- **Input:** From: noreply@rewards.airasia.com, Subject: Win a FREE trip to Singapore
  **Output:** {{"category": "Newsletter/Marketing", "urgency": "Low", "summary": "Promotional email about a contest to win a trip.", "action_item": null}}

Do not add any explanations. Only output the raw JSON object."""),
        ("user", """Here is the email to triage:
From: {from_sender}
Subject: {subject}

Body:
{body}

Now, provide the triage analysis as a JSON object.""")
    ])
    triage_chain = triage_prompt | llm

    ingestion_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a world-class text analysis expert. Your task is to process a given text and return a structured JSON object with two fields: 'summary' and 'tags'.

**GUIDELINES:**
- The 'summary' should be a concise, single-paragraph summary of the main points of the text.
- The 'tags' field should be a JSON array of 3-5 high-level, conceptual keywords that capture the core topics of the text.
- Do not add any explanations. Only output the raw JSON object.

**EXAMPLE:**
- **Input Text:** "The new study on quantum computing demonstrates a novel algorithm for prime factorization, potentially breaking current encryption standards. Researchers are excited but caution that practical implementation is still years away."
- **Output JSON:** {{"summary": "A recent study in quantum computing introduced a new prime factorization algorithm that could threaten existing encryption methods, though its practical use is not yet imminent.", "tags": ["quantum computing", "cryptography", "algorithm", "cybersecurity"]}}"""),
        ("user", "Here is the text to process:\n\n{text}\n\nNow, provide the analysis as a JSON object.")
    ])
    ingestion_chain = ingestion_prompt | llm


    print("✅ FastAPI server started successfully. LLM and chains are initialized.")

except Exception as e:
    print(f"❌ FATAL ERROR during initialization: {e}")
    chat_chain = None
    triage_chain = None
    ingestion_chain = None


# --- 4. API Endpoints ---

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    if chat_chain is None:
        return {"error": "LLM chat_chain is not initialized. Check server logs."}, 500

    print(f"Received chat query: {request.query}")
    response = chat_chain.invoke({"input": request.query})
    print("Generated response.")

    return {"response": response}


@app.post("/triage_email", response_model=TriageResponse)
def triage_email_endpoint(request: EmailTriageRequest):
    if triage_chain is None:
        return {"error": "LLM triage_chain is not initialized. Check server logs."}, 500

    print(f"Received triage request for email from: {request.from_sender}")

    soup = BeautifulSoup(request.body, "html.parser")
    clean_body_text = soup.get_text(separator='\\n', strip=True)

    response = triage_chain.invoke({
        "from_sender": request.from_sender,
        "subject": request.subject,
        "body": clean_body_text
    })

    try:
        triage_data = json.loads(response.content)
        print(f"Triage complete: {triage_data}")
        return triage_data
    except json.JSONDecodeError:
        print(f"❌ Error: LLM did not return valid JSON. Output was: {response.content}")
        return {"error": "Failed to parse LLM response as JSON."}, 500


@app.post("/memorize", response_model=IngestionResponse)
def memorize_endpoint(request: MemorizeRequest):
    """
    Processes raw text, generates a summary and keywords, and returns them.
    """
    if ingestion_chain is None:
        return {"error": "LLM ingestion_chain is not initialized. Check server logs."}, 500

    print(f"Received memorize request for text.")

    # 1. Get summary and conceptual tags from LLM
    try:
        llm_response = ingestion_chain.invoke({"text": request.text})
        ingestion_data = json.loads(llm_response.content)
        summary = ingestion_data.get("summary", "")
        llm_tags = ingestion_data.get("tags", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"❌ Error: LLM did not return valid JSON for memorize. Output was: {llm_response.content if 'llm_response' in locals() else 'N/A'}. Error: {e}")
        return {"error": "Failed to parse LLM response as JSON."}, 500

    # 2. Get statistical keywords from langextract
    # The 'find_features' function returns a dictionary of keywords and their scores.
    statistical_keywords = langextract.find_features(request.text)
    # We'll take the top 5 keywords.
    langextract_tags = list(statistical_keywords.keys())[:5]

    # 3. Combine and deduplicate tags
    combined_tags = list(set(llm_tags + langextract_tags))

    # 4. TODO: Add logic to create embedding and store in ChromaDB
    print(f"Memorization complete. Summary: {summary}, Tags: {combined_tags}")

    return {
        "summary": summary,
        "tags": combined_tags
    }


@app.post("/process_url", response_model=IngestionResponse)
def process_url_endpoint(request: ProcessUrlRequest):
    """
    Fetches content from a URL, generates a summary and keywords, and returns them.
    """
    if ingestion_chain is None:
        return {"error": "LLM ingestion_chain is not initialized. Check server logs."}, 500

    print(f"Received process_url request for: {request.url}")

    # 1. Fetch and parse URL content
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        # A simple approach to get text; might need refinement for complex sites
        clean_text = soup.get_text(separator='\\n', strip=True)
    except requests.RequestException as e:
        print(f"❌ Error fetching URL {request.url}: {e}")
        return {"error": f"Failed to fetch content from URL: {e}"}, 500

    if not clean_text:
        return {"error": "Could not extract any text content from the URL."}, 400

    # 2. Get summary and conceptual tags from LLM
    try:
        llm_response = ingestion_chain.invoke({"text": clean_text})
        ingestion_data = json.loads(llm_response.content)
        summary = ingestion_data.get("summary", "")
        llm_tags = ingestion_data.get("tags", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"❌ Error: LLM did not return valid JSON for process_url. Output was: {llm_response.content if 'llm_response' in locals() else 'N/A'}. Error: {e}")
        return {"error": "Failed to parse LLM response as JSON."}, 500

    # 3. Get statistical keywords from langextract
    statistical_keywords = langextract.find_features(clean_text)
    langextract_tags = list(statistical_keywords.keys())[:5]

    # 4. Combine and deduplicate tags
    combined_tags = list(set(llm_tags + langextract_tags))

    # 5. TODO: Add logic to create embedding and store in ChromaDB
    print(f"URL processing complete for: {request.url}")

    return {
        "summary": summary,
        "tags": combined_tags,
        "original_url": request.url
    }
