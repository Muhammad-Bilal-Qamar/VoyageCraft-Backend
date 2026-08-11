"""
VoyageCraft FastAPI backend.
Exposes endpoints to run the LangGraph multi-agent itinerary coordinator
and to simulate real-time disruptions.
"""

import os
import uuid
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from voyage_graph import run_initial_planning, run_disruption_simulation, VoyageState

app = FastAPI(
    title="VoyageCraft API",
    description="Dynamic Multi-Agent Itinerary Coordinator powered by LangGraph + Groq",
    version="1.0.0",
)

# CORS origins: localhost for local dev, plus whatever production frontend
# origin(s) you set via the FRONTEND_ORIGIN env var (comma-separated if you
# ever need more than one, e.g. a Vercel preview + production URL).
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_extra_origins = [
    o.strip() for o in os.getenv("FRONTEND_ORIGIN", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store only — no external DB. Trip plans live only for
# the lifetime of the server process/instance (and only in that instance's
# memory on Render, so they won't survive a redeploy or a free-tier spin
# down). That's expected: there's no Supabase/DB wired up, so nothing is
# persisted beyond the current session.
SESSIONS: Dict[str, VoyageState] = {}


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    origin_city: Optional[str] = Field(default="", examples=["Lahore, Pakistan"])
    destination: str = Field(..., min_length=1, examples=["Hunza Valley, Pakistan"])
    duration_days: int = Field(..., ge=1, examples=[5])  # no upper cap: long trips are generated in day-batches
    interests: List[str] = Field(..., min_length=1, examples=[["food", "history", "hiking"]])
    budget_total: float = Field(..., gt=0, examples=[1200])
    dietary_needs: Optional[List[str]] = Field(default_factory=list)
    travel_dates: Optional[str] = Field(default="")
    currency: str = Field(default="USD", examples=["USD"])


class DisruptionRequest(BaseModel):
    session_id: str
    description: str = Field(..., min_length=1, examples=["Severe thunderstorm warning"])
    time: str = Field(..., min_length=1, examples=["2:00 PM"])


class SessionResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]


def _sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Never echo the caller's Groq API key back in a response body."""
    clean = dict(state)
    clean.pop("groq_api_key", None)
    return clean


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
@app.head("/api/health")
def health_check(response: Response):
    """
    Health/uptime check. Supports both GET and HEAD explicitly so uptime
    monitors (e.g. UptimeRobot's HEAD-based "HTTP(s)" monitor type) get a
    clean 200 without needing to fetch/parse a body — this is what keeps a
    Render free-tier instance from spinning down due to inactivity.
    """
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    response.headers["Cache-Control"] = "no-store"
    return {"status": "ok", "groq_api_key_configured": groq_configured}


@app.post("/api/plan", response_model=SessionResponse)
def create_plan(
    req: PlanRequest,
    x_groq_api_key: Optional[str] = Header(default=None, alias="X-Groq-Api-Key"),
):
    groq_key = x_groq_api_key or os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(
            status_code=401,
            detail="No Groq API key provided. Add your Groq API key to continue.",
        )
    try:
        result_state = run_initial_planning(
            destination=req.destination,
            duration_days=req.duration_days,
            interests=req.interests,
            budget_total=req.budget_total,
            dietary_needs=req.dietary_needs,
            travel_dates=req.travel_dates or "",
            currency=req.currency or "USD",
            origin_city=req.origin_city or "",
            groq_api_key=groq_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Planning failed: {exc}")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = result_state

    return SessionResponse(session_id=session_id, state=_sanitize_state(result_state))


@app.get("/api/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(session_id=session_id, state=_sanitize_state(state))


@app.post("/api/disrupt", response_model=SessionResponse)
def simulate_disruption(
    req: DisruptionRequest,
    x_groq_api_key: Optional[str] = Header(default=None, alias="X-Groq-Api-Key"),
):
    state = SESSIONS.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    groq_key = x_groq_api_key or os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(
            status_code=401,
            detail="No Groq API key provided. Add your Groq API key to continue.",
        )
    try:
        updated_state = run_disruption_simulation(
            current_state=state,
            disruption_description=req.description,
            disruption_time=req.time,
            groq_api_key=groq_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Disruption handling failed: {exc}")

    SESSIONS[req.session_id] = updated_state
    return SessionResponse(session_id=req.session_id, state=_sanitize_state(updated_state))


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    # Render injects the port to bind to via the PORT env var.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
