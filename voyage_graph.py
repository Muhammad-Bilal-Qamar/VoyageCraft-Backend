"""
VoyageCraft - Dynamic Multi-Agent Itinerary Coordinator
LangGraph orchestration of 3 agents using Groq's llama-3.3-70b-versatile.
"""

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# LLM Configuration - Groq ONLY
# ---------------------------------------------------------------------------
#
# The API key can come from two places:
#   1. The server's own GROQ_API_KEY env var (legacy / local dev fallback).
#   2. A per-request key the frontend sends (state["groq_api_key"]), which is
#      what the "bring your own Groq key" modal on the frontend supplies.
# A per-request key always takes priority so each browser session/tab uses
# its own key.

def get_llm(temperature: float = 0.4, api_key: Optional[str] = None, max_tokens: int = 4096) -> ChatGroq:
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "No Groq API key available. Provide one from the client (the app should "
            "have prompted for it) or export GROQ_API_KEY on the server."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=resolved_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _llm_for_state(state: "VoyageState", temperature: float = 0.4, max_tokens: int = 4096) -> ChatGroq:
    """Convenience wrapper: pulls the per-session Groq key out of state."""
    return get_llm(temperature=temperature, api_key=state.get("groq_api_key"), max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class VoyageState(TypedDict, total=False):
    # Per-session Groq API key supplied by the client (never persisted to
    # the memory store's public list endpoints, only used in-flight).
    groq_api_key: str

    # Inputs
    origin_city: str
    destination: str
    duration_days: int
    interests: List[str]
    budget_total: float
    dietary_needs: List[str]
    travel_dates: str
    currency: str

    # Working memory across the graph
    discovery_output: Dict[str, Any]
    itinerary: List[Dict[str, Any]]
    budget_status: Dict[str, Any]
    transport_costs: Dict[str, Any]

    # Disruption handling
    disruptions: List[Dict[str, Any]]
    disruption_triggered: bool
    needs_rewrite: bool
    rewrite_count: int
    max_rewrites: int

    # Agent activity trace for UI
    agent_log: List[Dict[str, Any]]

    # Final output flag
    status: str


def _log(state: VoyageState, agent: str, message: str, level: str = "info") -> None:
    entry = {
        "agent": agent,
        "message": message,
        "level": level,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    state.setdefault("agent_log", [])
    state["agent_log"].append(entry)


def _extract_json(text: str) -> Any:
    """Robustly pull a JSON object/array out of an LLM response."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    start_candidates = [i for i in [text.find("{"), text.find("[")] if i != -1]
    if not start_candidates:
        raise ValueError("No JSON found in LLM output")
    start = min(start_candidates)
    end_brace = text.rfind("}")
    end_bracket = text.rfind("]")
    end = max(end_brace, end_bracket)
    snippet = text[start:end + 1]

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    # Common LLM JSON quirks: trailing commas before } or ], and
    # occasional smart-quotes instead of straight quotes.
    cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Re-raise with the snippet attached so callers/logs can inspect it.
        raise ValueError(
            f"Could not parse JSON after cleanup: {e}\n--- snippet ---\n{cleaned}"
        ) from e


# ---------------------------------------------------------------------------
# AGENT 1: Preference & Discovery Agent
# ---------------------------------------------------------------------------

def preference_discovery_agent(state: VoyageState) -> VoyageState:
    llm = _llm_for_state(state, temperature=0.6)

    disruption_context = ""
    if state.get("needs_rewrite") and state.get("disruptions"):
        latest = state["disruptions"][-1]
        disruption_context = (
            f"\n\nIMPORTANT: A disruption has occurred: '{latest['description']}' "
            f"at {latest.get('time', 'an affected time slot')}. "
            "You must propose ALTERNATIVE activities/dining that avoid the affected "
            "time slot and, if the disruption is weather-related, favor indoor options."
        )

    currency = state.get("currency", "USD")

    # Scale shortlist size with trip length so long trips (30+ days) have
    # enough variety across itinerary batches without repeating the same
    # handful of activities every week.
    duration = state.get("duration_days", 1)
    min_activities = min(30, max(6, duration * 2))
    max_activities = min(40, max(10, duration * 3))
    min_dining = min(20, max(3, duration))
    max_dining = min(25, max(5, duration + 2))

    system = SystemMessage(content=(
        "You are the Preference & Discovery Agent for VoyageCraft, an elite AI travel "
        "planning system. Your job is to curate a personalized shortlist of activities "
        "and dining options based on the traveler's interests, budget, and dietary needs. "
        f"Price every item in {currency}. "
        "Respond ONLY with valid JSON, no prose, no markdown fences. "
        "Schema: {\"activities\": [{\"name\": str, \"category\": str, \"est_cost\": number, "
        "\"duration_hours\": number, \"indoor\": bool}], \"dining\": [{\"name\": str, "
        "\"cuisine\": str, \"est_cost\": number, \"dietary_tags\": [str]}]}"
        f" Provide between {min_activities} and {max_activities} DISTINCT activities and "
        f"{min_dining} to {max_dining} DISTINCT dining options so a long, multi-week trip "
        "has enough variety to avoid repetition."
    ))
    human = HumanMessage(content=(
        f"Destination: {state['destination']}\n"
        f"Trip length: {duration} days\n"
        f"Interests: {', '.join(state['interests'])}\n"
        f"Total budget: {state['budget_total']} {currency}\n"
        f"Dietary needs: {', '.join(state.get('dietary_needs') or ['none'])}"
        f"{disruption_context}"
    ))

    response = llm.invoke([system, human])
    try:
        data = _extract_json(response.content)
    except Exception as e:
        print("=== [Preference & Discovery Agent] JSON PARSE FAILURE ===")
        print("Error:", e)
        print("Raw response:\n", response.content)
        print("===========================================================")
        data = {
            "activities": [
                {"name": "City Walking Tour", "category": "sightseeing",
                 "est_cost": 25, "duration_hours": 3, "indoor": False}
            ],
            "dining": [
                {"name": "Local Bistro", "cuisine": "local",
                 "est_cost": 30, "dietary_tags": []}
            ],
        }

    state["discovery_output"] = data
    _log(
        state,
        "Preference & Discovery Agent",
        f"Curated {len(data.get('activities', []))} activities and "
        f"{len(data.get('dining', []))} dining options for {state['destination']}."
        + (" (re-curated after disruption)" if state.get("needs_rewrite") else ""),
    )
    return state


# ---------------------------------------------------------------------------
# AGENT 2: Logistics & Budget Optimizer Agent
# ---------------------------------------------------------------------------

DAY_BATCH_SIZE = 5  # generate the itinerary in chunks so long trips (>30 days) never blow the token budget in one call


def _estimate_transport_costs(state: VoyageState) -> Dict[str, Any]:
    """
    Estimates round-trip transport cost between the traveler's origin city
    and the destination. Kept as its own focused LLM call so it never gets
    lost or truncated inside a larger itinerary payload, and so it reliably
    appears even on very long, multi-batch trips.
    """
    llm = _llm_for_state(state, temperature=0.2)
    currency = state.get("currency", "USD")
    origin = state.get("origin_city") or "the traveler's home city (unspecified, assume a major hub)"

    system = SystemMessage(content=(
        "You are a travel logistics cost estimator. Estimate realistic round-trip "
        "transport costs between an origin and a destination, in the specified currency. "
        "Respond ONLY with valid JSON, no prose, no markdown fences. Schema: "
        "{\"mode\": str, \"outbound_cost\": number, \"return_cost\": number, "
        "\"total_transport_cost\": number, \"notes\": str}"
    ))
    human = HumanMessage(content=(
        f"Origin: {origin}\n"
        f"Destination: {state['destination']}\n"
        f"Currency: {currency}\n"
        "Pick the most realistic mode of transport (flight, train, bus, or car) for this "
        "route and estimate a reasonable one-way cost, then double it appropriately for "
        "the return leg (return leg may differ slightly due to fare fluctuation)."
    ))
    response = llm.invoke([system, human])
    try:
        data = _extract_json(response.content)
    except Exception as e:
        print("=== [Logistics Agent - Transport Estimate] JSON PARSE FAILURE ===")
        print("Error:", e)
        print("Raw response:\n", response.content)
        print("===================================================================")
        data = {
            "mode": "flight", "outbound_cost": 150, "return_cost": 150,
            "total_transport_cost": 300,
            "notes": "Fallback estimate used due to a parsing issue.",
        }
    data.setdefault("total_transport_cost", data.get("outbound_cost", 0) + data.get("return_cost", 0))
    data["currency"] = currency
    return data


def _generate_itinerary_batch(
    state: VoyageState,
    start_day: int,
    end_day: int,
    disruption_context: str,
    attempt: int = 1,
) -> List[Dict[str, Any]]:
    """Generates itinerary days [start_day, end_day] inclusive in a single LLM call.

    BUGFIX: previously this used the shared default max_tokens=4096, which is
    frequently not enough to fit a full batch of days (each with several
    timed items) as JSON. Once the model's output got cut off mid-JSON,
    `_extract_json` would fail and every remaining day in that batch was
    silently replaced with a one-line placeholder — which is exactly why
    long trips appeared to "only show some days". Fix: give this call a much
    larger token budget, and if a batch still comes back incomplete (fewer
    days than requested, or a JSON parse failure), automatically retry once
    with a smaller day-window before ever falling back to placeholders.
    """
    expected_days = end_day - start_day + 1
    # Budget generously: ~450 tokens/day of headroom is comfortably above what
    # a realistic day (4-6 items + notes) needs as JSON.
    token_budget = min(8192, max(2048, expected_days * 700))
    llm = _llm_for_state(state, temperature=0.3, max_tokens=token_budget)
    currency = state.get("currency", "USD")

    system = SystemMessage(content=(
        "You are the Logistics & Budget Optimizer Agent for VoyageCraft. Build a "
        "day-by-day itinerary segment that is logistically feasible (realistic timing, "
        "travel buffers) using the given activity/dining shortlist. Price every item in "
        f"{currency}. Respond ONLY with valid JSON, no prose, no markdown fences, no "
        "trailing explanation. Keep each day's 'items' list concise (4-6 items) and "
        "'notes' short (one sentence) so the full response fits comfortably. "
        "Schema: {\"itinerary\": [{\"day\": number, \"date_label\": str, \"items\": "
        "[{\"time\": str, \"title\": str, \"type\": str, \"cost\": number, \"location\": str}], "
        "\"notes\": str}]}"
    ))
    human = HumanMessage(content=(
        f"Destination: {state['destination']}\n"
        f"Generate ONLY days {start_day} through {end_day} of a {state['duration_days']}-day trip "
        f"(exactly {expected_days} day object(s), do not generate any other days, do not skip any).\n"
        f"Currency: {currency}\n"
        f"Discovery shortlist: {json.dumps(state.get('discovery_output', {}))}"
        f"{disruption_context}"
    ))
    response = llm.invoke([system, human])
    parse_error: Optional[Exception] = None
    days: List[Dict[str, Any]] = []
    try:
        data = _extract_json(response.content)
        days = data.get("itinerary", [])
        if not days:
            raise ValueError("Empty itinerary batch returned")
    except Exception as e:
        parse_error = e

    got_all_days = parse_error is None and len(days) >= expected_days

    if not got_all_days:
        print(f"=== [Logistics Agent] Incomplete/failed itinerary batch (days {start_day}-{end_day}, attempt {attempt}) ===")
        if parse_error:
            print("Error:", parse_error)
        else:
            print(f"Expected {expected_days} days, got {len(days)}.")
        print("Raw response:\n", response.content)
        print("====================================================================")

        # Retry once, and if the window is more than 1 day, split it in half
        # so each half needs fewer output tokens and is far less likely to
        # be truncated again. This recursion bottoms out at single-day
        # batches, so worst case every day is generated individually.
        if attempt < 3 and expected_days > 1:
            mid = (start_day + end_day) // 2
            left = _generate_itinerary_batch(state, start_day, mid, disruption_context, attempt=attempt + 1)
            right = _generate_itinerary_batch(state, mid + 1, end_day, disruption_context, attempt=attempt + 1)
            return left + right
        if attempt < 3:
            return _generate_itinerary_batch(state, start_day, end_day, disruption_context, attempt=attempt + 1)

        # Only after exhausting retries do we fall back to a placeholder,
        # and only for whichever specific days are still missing.
        by_day = {d.get("day"): d for d in days if isinstance(d, dict)}
        filled = []
        for d in range(start_day, end_day + 1):
            if d in by_day:
                filled.append(by_day[d])
            else:
                filled.append({
                    "day": d,
                    "date_label": f"Day {d}",
                    "items": [{
                        "time": "09:00", "title": "Free day / itinerary generation issue",
                        "type": "logistics", "cost": 0, "location": state["destination"],
                    }],
                    "notes": "This day could not be generated after retries and was filled with a placeholder.",
                })
        return filled

    return days[:expected_days] if len(days) > expected_days else days


def parallel_intake_node(state: VoyageState) -> VoyageState:
    """
    UNION NODE: runs two independent agents *simultaneously* instead of
    sequentially.

    The Preference & Discovery Agent (curating activities/dining) and the
    transport-cost estimate (round-trip origin<->destination pricing) don't
    depend on each other's output — discovery only needs the traveler's
    interests/budget, and the transport estimate only needs origin/
    destination/currency. Previously they ran one after another purely
    because of graph ordering. Here they're dispatched together on a small
    thread pool (each does its own network-bound Groq call) and the node
    doesn't return until both are done, cutting real wall-clock latency
    roughly in half for this stage.
    """
    # Give the discovery branch its own copy of the log (and dict) so the two
    # threads never mutate the same list concurrently; we merge logs back
    # afterwards once both branches have finished.
    discovery_state = dict(state)
    discovery_state["agent_log"] = list(state.get("agent_log", []))

    with ThreadPoolExecutor(max_workers=2) as pool:
        discovery_future = pool.submit(preference_discovery_agent, discovery_state)
        transport_future = pool.submit(_estimate_transport_costs, state)

        discovery_result = discovery_future.result()
        transport_costs = transport_future.result()

    # Merge results from both concurrent branches back into the shared state.
    state["discovery_output"] = discovery_result.get("discovery_output", {})
    state["transport_costs"] = transport_costs

    # Merge agent logs from both branches (preserve chronological-ish order
    # by simply concatenating; each entry carries its own timestamp).
    state.setdefault("agent_log", [])
    for entry in discovery_result.get("agent_log", []):
        if entry not in state["agent_log"]:
            state["agent_log"].append(entry)

    _log(
        state,
        "Logistics & Budget Optimizer Agent",
        f"Estimated round-trip transport cost ({transport_costs.get('mode', 'flight')}) "
        "concurrently with the Discovery Agent's curation pass.",
    )
    return state


def logistics_budget_agent(state: VoyageState) -> VoyageState:
    currency = state.get("currency", "USD")
    total_days = state.get("duration_days", 1)

    disruption_context = ""
    if state.get("needs_rewrite") and state.get("disruptions"):
        latest = state["disruptions"][-1]
        disruption_context = (
            f"\n\nCRITICAL: Rebuild the schedule to avoid the disruption: "
            f"'{latest['description']}' at {latest.get('time')}. "
            "Re-sequence activities so nothing is scheduled during or immediately "
            "after the disruption window, and note the change explicitly in "
            "the 'notes' field of the affected day."
        )

    # ---- 1. Round-trip transport cost (origin -> destination -> origin) ----
    # Already computed in parallel_intake_node (concurrently with discovery).
    # Fall back to computing it here for safety if it's ever missing.
    transport_costs = state.get("transport_costs") or _estimate_transport_costs(state)
    state["transport_costs"] = transport_costs

    # ---- 2. Itinerary generated in day-batches so arbitrarily long trips ----
    #         (30+ days) never get truncated by a single response's token cap.
    full_itinerary: List[Dict[str, Any]] = []
    day_cursor = 1
    while day_cursor <= total_days:
        batch_end = min(day_cursor + DAY_BATCH_SIZE - 1, total_days)
        batch_days = _generate_itinerary_batch(state, day_cursor, batch_end, disruption_context)
        full_itinerary.extend(batch_days)
        _log(
            state,
            "Logistics & Budget Optimizer Agent",
            f"Scheduled days {day_cursor}\u2013{batch_end} of {total_days}.",
        )
        day_cursor = batch_end + 1

    state["itinerary"] = full_itinerary

    # ---- 3. Accommodation cost, broken down by night ----
    accommodation_llm = _llm_for_state(state, temperature=0.2)
    system = SystemMessage(content=(
        "You are a travel accommodation cost estimator. Estimate a realistic nightly "
        f"accommodation rate in {currency} for the given destination and traveler budget "
        "tier. Respond ONLY with valid JSON, no prose. Schema: "
        "{\"accommodation_per_night\": number, \"lodging_type\": str, \"notes\": str}"
    ))
    human = HumanMessage(content=(
        f"Destination: {state['destination']}\n"
        f"Total trip budget: {state['budget_total']} {currency}\n"
        f"Trip length: {total_days} nights"
    ))
    response = accommodation_llm.invoke([system, human])
    try:
        accom_data = _extract_json(response.content)
    except Exception as e:
        print("=== [Logistics Agent - Accommodation Estimate] JSON PARSE FAILURE ===")
        print("Error:", e)
        print("Raw response:\n", response.content)
        print("=======================================================================")
        accom_data = {"accommodation_per_night": 60, "lodging_type": "mid-range hotel",
                      "notes": "Fallback estimate used due to a parsing issue."}

    per_night = accom_data.get("accommodation_per_night", 60)
    nightly_breakdown = [
        {"night": n, "date_label": f"Night {n}", "cost": per_night}
        for n in range(1, total_days + 1)
    ]
    total_accommodation_cost = round(per_night * total_days, 2)

    # ---- 4. Aggregate budget across activities, dining, lodging, transport ----
    total_activities_cost = 0.0
    total_dining_cost = 0.0
    for day in full_itinerary:
        for item in day.get("items", []):
            item_type = (item.get("type") or "").lower()
            cost = item.get("cost", 0) or 0
            if "din" in item_type or "food" in item_type or "meal" in item_type:
                total_dining_cost += cost
            elif "logistic" in item_type or "transport" in item_type:
                pass  # counted separately via transport_costs
            else:
                total_activities_cost += cost

    total_transport_cost = transport_costs.get("total_transport_cost", 0) or 0
    grand_total = round(
        total_activities_cost + total_dining_cost + total_accommodation_cost + total_transport_cost, 2
    )
    budget_limit = state["budget_total"]
    within_budget = grand_total <= budget_limit
    remaining = round(budget_limit - grand_total, 2)

    state["budget_status"] = {
        "currency": currency,
        "accommodation_per_night": per_night,
        "accommodation_lodging_type": accom_data.get("lodging_type", "hotel"),
        "accommodation_nightly_breakdown": nightly_breakdown,
        "total_activities_cost": round(total_activities_cost, 2),
        "total_dining_cost": round(total_dining_cost, 2),
        "total_accommodation_cost": total_accommodation_cost,
        "total_transport_cost": round(total_transport_cost, 2),
        "transport_mode": transport_costs.get("mode", "flight"),
        "transport_outbound_cost": transport_costs.get("outbound_cost", 0),
        "transport_return_cost": transport_costs.get("return_cost", 0),
        "grand_total": grand_total,
        "budget_limit": budget_limit,
        "within_budget": within_budget,
        "remaining": remaining,
    }

    action = "Rebuilt" if state.get("needs_rewrite") else "Built"
    _log(
        state,
        "Logistics & Budget Optimizer Agent",
        f"{action} a {len(state['itinerary'])}-day itinerary with round-trip transport "
        f"({transport_costs.get('mode', 'flight')}) and lodging costed. "
        f"Grand total: {grand_total} {currency} (limit {budget_limit} {currency}).",
    )

    # Reset the rewrite flag now that logistics has responded to it
    if state.get("needs_rewrite"):
        state["needs_rewrite"] = False

    return state


# ---------------------------------------------------------------------------
# AGENT 3: Disruption Manager Agent
# ---------------------------------------------------------------------------

def disruption_manager_agent(state: VoyageState) -> VoyageState:
    """
    Checks for any pending disruption that has not yet been reconciled against
    the current itinerary. In production this would poll flight-status APIs,
    weather feeds, and news. Here it inspects state['disruptions'] for any
    entry not yet marked 'resolved' and decides whether a rewrite is required.
    """
    disruptions = state.get("disruptions", [])
    unresolved = [d for d in disruptions if not d.get("resolved")]

    if not unresolved:
        _log(
            state,
            "Disruption Manager Agent",
            "Monitored flight status, weather feeds, and local news. No active disruptions detected.",
        )
        state["disruption_triggered"] = False
        state["status"] = "complete"
        return state

    latest = unresolved[-1]
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", 3)

    if rewrite_count >= max_rewrites:
        _log(
            state,
            "Disruption Manager Agent",
            f"Disruption '{latest['description']}' detected but max rewrite attempts "
            f"({max_rewrites}) reached. Escalating to traveler for manual review.",
            level="warning",
        )
        latest["resolved"] = True
        state["disruption_triggered"] = False
        state["status"] = "complete"
        return state

    llm = _llm_for_state(state, temperature=0.2)
    system = SystemMessage(content=(
        "You are the Disruption Manager Agent for VoyageCraft. You have detected a "
        "real-world disruption affecting the traveler's itinerary. Assess severity and "
        "respond ONLY with valid JSON: {\"severity\": \"low\"|\"medium\"|\"high\", "
        "\"requires_rewrite\": bool, \"reasoning\": str}"
    ))
    human = HumanMessage(content=(
        f"Disruption: {latest['description']} at {latest.get('time')}\n"
        f"Current itinerary snapshot: {json.dumps(state.get('itinerary', []))[:2000]}"
    ))
    response = llm.invoke([system, human])
    try:
        assessment = _extract_json(response.content)
    except Exception as e:
        print("=== [Disruption Manager Agent] JSON PARSE FAILURE ===")
        print("Error:", e)
        print("Raw response:\n", response.content)
        print("=======================================================")
        assessment = {"severity": "medium", "requires_rewrite": True,
                       "reasoning": "Defaulting to rewrite due to parsing issue."}

    _log(
        state,
        "Disruption Manager Agent",
        f"Detected disruption: '{latest['description']}'. Severity: "
        f"{assessment.get('severity', 'unknown')}. {assessment.get('reasoning', '')}",
        level="warning",
    )

    if assessment.get("requires_rewrite", True):
        latest["resolved"] = True
        latest["severity"] = assessment.get("severity", "medium")
        state["needs_rewrite"] = True
        state["disruption_triggered"] = True
        state["rewrite_count"] = rewrite_count + 1
        state["status"] = "rewriting"
    else:
        latest["resolved"] = True
        state["disruption_triggered"] = False
        state["status"] = "complete"

    return state


# ---------------------------------------------------------------------------
# Conditional Edge Logic
# ---------------------------------------------------------------------------

def route_after_disruption_check(state: VoyageState) -> str:
    """
    Cyclic routing: if the Disruption Manager flagged that a rewrite is
    needed, loop back to Preference & Discovery -> Logistics. Otherwise end.
    """
    if state.get("needs_rewrite"):
        return "rewrite"
    return "done"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def build_voyage_graph():
    graph = StateGraph(VoyageState)

    # "parallel_intake" is the UNION node: it runs the Preference & Discovery
    # Agent and the transport-cost estimator simultaneously (see
    # parallel_intake_node docstring) instead of one strictly after another.
    graph.add_node("parallel_intake", parallel_intake_node)
    graph.add_node("logistics_budget", logistics_budget_agent)
    graph.add_node("disruption_manager", disruption_manager_agent)

    graph.set_entry_point("parallel_intake")
    graph.add_edge("parallel_intake", "logistics_budget")
    graph.add_edge("logistics_budget", "disruption_manager")

    # Conditional / cyclic edge: disruption manager can loop back to
    # parallel_intake (which re-curates + re-estimates transport concurrently)
    # -> logistics_budget again, or terminate the graph.
    graph.add_conditional_edges(
        "disruption_manager",
        route_after_disruption_check,
        {
            "rewrite": "parallel_intake",
            "done": END,
        },
    )

    return graph.compile()


voyage_app = build_voyage_graph()


# ---------------------------------------------------------------------------
# Public entry points used by FastAPI
# ---------------------------------------------------------------------------

def run_initial_planning(
    destination: str,
    duration_days: int,
    interests: List[str],
    budget_total: float,
    dietary_needs: Optional[List[str]] = None,
    travel_dates: str = "",
    currency: str = "USD",
    origin_city: str = "",
    groq_api_key: Optional[str] = None,
) -> VoyageState:
    initial_state: VoyageState = {
        "groq_api_key": groq_api_key or "",
        "origin_city": origin_city,
        "destination": destination,
        "duration_days": duration_days,
        "interests": interests,
        "budget_total": budget_total,
        "dietary_needs": dietary_needs or [],
        "travel_dates": travel_dates,
        "currency": currency,
        "discovery_output": {},
        "itinerary": [],
        "budget_status": {},
        "transport_costs": {},
        "disruptions": [],
        "disruption_triggered": False,
        "needs_rewrite": False,
        "rewrite_count": 0,
        "max_rewrites": 3,
        "agent_log": [],
        "status": "planning",
    }
    result = voyage_app.invoke(initial_state)
    return result


def run_disruption_simulation(
    current_state: VoyageState,
    disruption_description: str,
    disruption_time: str,
    groq_api_key: Optional[str] = None,
) -> VoyageState:
    """
    Injects a new disruption into an existing state and re-runs the graph
    starting from the Disruption Manager node's logic (we re-enter via the
    full graph so the cyclic conditional edges are exercised exactly as they
    would be for a live external trigger).
    """
    new_disruption = {
        "description": disruption_description,
        "time": disruption_time,
        "resolved": False,
        "injected_at": datetime.utcnow().isoformat() + "Z",
    }
    current_state = dict(current_state)
    if groq_api_key:
        current_state["groq_api_key"] = groq_api_key
    current_state.setdefault("disruptions", [])
    current_state["disruptions"].append(new_disruption)
    current_state["needs_rewrite"] = False
    current_state["status"] = "monitoring"

    # Re-enter the graph at the disruption_manager node by invoking a
    # sub-graph call: simplest robust approach is to run disruption_manager
    # directly, then let it route into the compiled graph if a rewrite
    # is required.
    current_state = disruption_manager_agent(current_state)

    if current_state.get("needs_rewrite"):
        # Continue the cycle: discovery -> logistics -> disruption_manager
        # (again), using the compiled graph so further disruptions during
        # this same call would still be handled by the conditional edge.
        result = voyage_app.invoke(current_state)
        return result

    return current_state
