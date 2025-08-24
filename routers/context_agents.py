import json
from fastapi import APIRouter

from schemas import (
    TransactionRequest, TransactionResponse,
    WellnessSummaryRequest, WellnessSummaryResponse,
    HomeEventRequest, HomeEventResponse,
    CalendarEventRequest, CalendarEventResponse,
    SynthesisRequest, SynthesisResponse
)
from chains import (
    transaction_chain, wellness_chain,
    home_event_chain, calendar_event_chain,
    synthesis_chain
)

router = APIRouter(
    prefix="/context",
    tags=["Contextual Agents"]
)

@router.post("/log_transaction", response_model=TransactionResponse)
def log_transaction_endpoint(request: TransactionRequest):
    if transaction_chain is None:
        return {"error": "LLM transaction_chain is not initialized."}, 500

    response = transaction_chain.invoke({"text": request.text})
    try:
        data = json.loads(response.content)
        # TODO: Add logic to store this structured transaction data in PostgreSQL.
        return data
    except json.JSONDecodeError:
        return {"error": f"Failed to parse LLM response as JSON. Output: {response.content}"}, 500

@router.post("/wellness_summary", response_model=WellnessSummaryResponse)
def wellness_summary_endpoint(request: WellnessSummaryRequest):
    if wellness_chain is None:
        return {"error": "LLM wellness_chain is not initialized."}, 500

    summary_text = wellness_chain.invoke({
        "calendar_events": request.calendar_events,
        "fit_data": json.dumps(request.fit_data) # Ensure dict is passed as a string
    })
    return {"summary_text": summary_text}

@router.post("/narrate_home_event", response_model=HomeEventResponse)
def narrate_home_event_endpoint(request: HomeEventRequest):
    if home_event_chain is None:
        return {"error": "LLM home_event_chain is not initialized."}, 500

    response = home_event_chain.invoke({
        "primary_event": json.dumps(request.primary_event),
        "context": json.dumps(request.context)
    })
    try:
        data = json.loads(response.content)
        # TODO: Add logic to log all narratives to PostgreSQL.
        return data
    except json.JSONDecodeError:
        return {"error": f"Failed to parse LLM response as JSON. Output: {response.content}"}, 500

@router.post("/process_calendar_event", response_model=CalendarEventResponse)
def process_calendar_event_endpoint(request: CalendarEventRequest):
    if calendar_event_chain is None:
        return {"error": "LLM calendar_event_chain is not initialized."}, 500

    # Pass the entire event model as a JSON string for the prompt
    event_details = request.model_dump_json()

    response = calendar_event_chain.invoke({"event_details": event_details})
    try:
        data = json.loads(response.content)
        # TODO: Add logic to store structured event data in PostgreSQL.
        return {
            "category": data.get("category", "Other"),
            "original_event": request
        }
    except json.JSONDecodeError:
        return {"error": f"Failed to parse LLM response as JSON. Output: {response.content}"}, 500

@router.post("/synthesize_report", response_model=SynthesisResponse)
def synthesize_report_endpoint(request: SynthesisRequest):
    if synthesis_chain is None:
        return {"error": "LLM synthesis_chain is not initialized."}, 500

    data_as_string = json.dumps(request.data, indent=2)

    report_text = synthesis_chain.invoke({
        "report_type": request.report_type,
        "data_as_string": data_as_string
    })

    return {"report_text": report_text}
