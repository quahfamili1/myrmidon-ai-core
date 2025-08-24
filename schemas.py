from pydantic import BaseModel
from typing import Optional, List

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


# --- Group 1: Historical & Interest Data Ingestion ---

class TakeoutRequest(BaseModel):
    source_type: str
    content: str

class TakeoutResponse(BaseModel):
    topic: str
    summary: str
    tags: list[str]


# --- Group 2: Real-World Context Agents ---

class TransactionRequest(BaseModel):
    text: str

class TransactionResponse(BaseModel):
    merchant_name: str
    amount: float
    currency: str
    category: str

class WellnessSummaryRequest(BaseModel):
    calendar_events: list[str]
    fit_data: dict

class WellnessSummaryResponse(BaseModel):
    summary_text: str

class HomeEventRequest(BaseModel):
    primary_event: dict
    context: dict

class HomeEventResponse(BaseModel):
    narrative: str
    significance: int

class CalendarEventRequest(BaseModel):
    summary: str
    description: Optional[str] = None
    start_time: str
    end_time: str
    location: Optional[str] = None
    attendees: Optional[list[str]] = None

class CalendarEventResponse(BaseModel):
    category: str
    original_event: CalendarEventRequest

class SynthesisRequest(BaseModel):
    report_type: str
    data: List[dict]

class SynthesisResponse(BaseModel):
    report_text: str
