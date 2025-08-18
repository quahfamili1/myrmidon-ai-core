import json
import requests
import langextract
from fastapi import APIRouter
from bs4 import BeautifulSoup

from schemas import (
    QueryRequest,
    EmailTriageRequest,
    TriageResponse,
    MemorizeRequest,
    ProcessUrlRequest,
    IngestionResponse,
    TakeoutRequest,
    TakeoutResponse
)
from chains import chat_chain, triage_chain, ingestion_chain, takeout_chain

router = APIRouter(
    prefix="/ingestion",
    tags=["Ingestion Agents"]
)

@router.post("/chat")
def chat_endpoint(request: QueryRequest):
    if chat_chain is None:
        return {"error": "LLM chat_chain is not initialized. Check server logs."}, 500

    print(f"Received chat query: {request.query}")
    response = chat_chain.invoke({"input": request.query})
    print("Generated response.")

    return {"response": response}


@router.post("/triage_email", response_model=TriageResponse)
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


@router.post("/memorize", response_model=IngestionResponse)
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
    statistical_keywords = langextract.find_features(request.text)
    langextract_tags = list(statistical_keywords.keys())[:5]

    # 3. Combine and deduplicate tags
    combined_tags = list(set(llm_tags + langextract_tags))

    # 4. TODO: Add logic to create embedding and store in ChromaDB
    print(f"Memorization complete. Summary: {summary}, Tags: {combined_tags}")

    return {
        "summary": summary,
        "tags": combined_tags
    }


@router.post("/process_url", response_model=IngestionResponse)
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
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
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


@router.post("/process_takeout_data", response_model=TakeoutResponse)
def process_takeout_data_endpoint(request: TakeoutRequest):
    """
    Processes historical data from Google Takeout (Gemini or Chrome).
    """
    if takeout_chain is None:
        return {"error": "LLM takeout_chain is not initialized."}, 500

    print(f"Received takeout request for source: {request.source_type}")

    # 1. Get topic, summary, and conceptual tags from LLM
    try:
        llm_response = takeout_chain.invoke({
            "source_type": request.source_type,
            "content": request.content
        })
        ingestion_data = json.loads(llm_response.content)
        topic = ingestion_data.get("topic", "Unknown")
        summary = ingestion_data.get("summary", "")
        llm_tags = ingestion_data.get("tags", [])
    except (json.JSONDecodeError, AttributeError) as e:
        return {"error": f"Failed to parse LLM response as JSON. Output: {llm_response.content if 'llm_response' in locals() else 'N/A'}"}, 500

    # 2. Get statistical keywords from langextract
    statistical_keywords = langextract.find_features(request.content)
    langextract_tags = list(statistical_keywords.keys())[:5]

    # 3. Combine and deduplicate tags
    combined_tags = list(set(llm_tags + langextract_tags))

    # 4. TODO: Add logic to create embedding from the summary and store in ChromaDB.
    print(f"Takeout processing complete for source: {request.source_type}")

    return {
        "topic": topic,
        "summary": summary,
        "tags": combined_tags
    }
