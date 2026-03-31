import os
import json
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import requests
from bs4 import BeautifulSoup
import re

from mocks import search_knowledge_base
from database import init_db, get_db, Lead


load_dotenv()

app = FastAPI(title="Command Center API", version="1.0.0")

@app.on_event("startup")
def on_startup():
    init_db()

# --- SSE Clients ---
sse_clients = set()

async def broadcast_lead(lead_data: dict):
    for q in list(sse_clients):
        await q.put(lead_data)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app", # allow vercel preview deployments
        "https://faroukhajjej.com", # assuming this is the production domain
        "*" # Fallback for now, but with credentials=False it works
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Definition ---
class AgentState(TypedDict):
    messages: list[Any]
    current_agent: str
    kb_results: str
    final_script: str
    log_stream: list[Dict[str, str]]
    company_domain: str

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Nodes (The Agents) ---

async def triage_agent(state: AgentState):
    """Analyzes the initial request and decides next steps."""
    user_request = state["messages"][0].content
    
    log = {"agent": "Triage AI", "action": "Analyzing incoming IT support ticket...", "status": "running"}
    state["log_stream"].append(log)
    
    try:
        prompt = f"Analyze this request to determine if it is an IT support ticket or a request for a Sales/AI Audit of a company domain. Request: '{user_request}'. Respond with EXACTLY ONE word: either 'IT_SUPPORT' or 'SALES_LEAD'."
        response = await llm.ainvoke(prompt)
        intent = response.content.strip().replace("\n", "").upper()
        
        if "SALES_LEAD" in intent:
            intent = "SALES_LEAD"
            next_agent = "sourcing_agent"
            log = {"agent": "Triage AI", "action": f"Identified intent: SALES_LEAD. Routing to Sourcing Agent.", "status": "success"}
        else:
            intent = "IT_SUPPORT"
            next_agent = "retrieval_agent"
            log = {"agent": "Triage AI", "action": f"Identified intent: IT_SUPPORT. Routing to Retrieval Agent.", "status": "success"}

        state["log_stream"].append(log)
        state["current_agent"] = next_agent
        state["messages"].append(AIMessage(content=f"Triage complete. Intent: {intent}"))
    except Exception as e:
        error_msg = str(e)
        log = {"agent": "Triage AI", "action": f"Error analyzing request: {error_msg[:100]}... Routing to Fallback.", "status": "error"}
        state["log_stream"].append(log)
        
        # Fallback intent if LLM fails
        state["current_agent"] = "retrieval_agent"
        state["messages"].append(AIMessage(content="Triage failed due to API error. Proceeding to retrieval as fallback."))
        
    return state

async def retrieval_agent(state: AgentState):
    """Searches the internal knowledge base for SOPs."""
    user_request = state["messages"][0].content
    
    log = {"agent": "RAG Agent", "action": "Querying Enterprise VectorDB for Standard Operating Procedures...", "status": "running"}
    state["log_stream"].append(log)
    
    try:
        results = search_knowledge_base(user_request)
        state["kb_results"] = results
        
        log = {"agent": "RAG Agent", "action": f"Found {len(results.split('---'))} matching documents in IT Knowledge Base.", "status": "success"}
        state["log_stream"].append(log)
    except Exception as e:
        error_msg = str(e)
        log = {"agent": "RAG Agent", "action": f"Vector retrieval failed: {error_msg}. Using empty context.", "status": "error"}
        state["log_stream"].append(log)
        state["kb_results"] = "No relevant documentation found due to a system error."
    
    state["current_agent"] = "action_agent"
    state["messages"].append(AIMessage(content=f"Retrieval complete. Results: {state['kb_results'][:100]}..."))
    return state

async def action_agent(state: AgentState):
    """Drafts the PowerShell or Bash script based on the SOP."""
    
    user_request = state["messages"][0].content
    kb_content = state["kb_results"]
    
    log = {"agent": "Action AI", "action": "Drafting remediation script based on SOP...", "status": "running"}
    state["log_stream"].append(log)
    
    try:
        prompt = f"Write a PowerShell or Bash remediation script to resolve the following IT ticket: '{user_request}'. Use this Standard Operating Procedure as context: '{kb_content}'. Provide ONLY the script code. Do not wrap code in markdown formatting."
        response = await llm.ainvoke(prompt)
        
        script = response.content.strip()
        if script.startswith("```"):
            lines = script.split("\n")
            if len(lines) > 2:
                script = "\n".join(lines[1:-1])

        state["final_script"] = script
        
        log = {"agent": "Action AI", "action": "Script generation complete. Ready for human review.", "status": "success"}
        state["log_stream"].append(log)
    except Exception as e:
        error_msg = str(e)
        log = {"agent": "Action AI", "action": f"Script generation failed: {error_msg}. Escalation required.", "status": "error"}
        state["log_stream"].append(log)
        state["final_script"] = f"# ERROR: AI Generation Failed.\n# Reason: {error_msg}\n# Escalation: Please route to a human L3 engineer."
        
    state["current_agent"] = "end"
    state["messages"].append(AIMessage(content="Action plan generated."))
    return state

async def sourcing_agent(state: AgentState):
    """Scrapes the target domain for the Sales Lead flow."""
    user_request = state["messages"][0].content
    
    # Extract domain using simple regex
    domain_match = re.search(r'[\w.-]+\.\w+', user_request)
    domain_str = domain_match.group(0) if domain_match else "example.com"
    state["company_domain"] = domain_str
    
    log = {"agent": "Sourcing Agent", "action": f"Analyzing target domain: {domain_str}...", "status": "running"}
    state["log_stream"].append(log)
    
    try:
        url = f"https://{domain_str}" if not domain_str.startswith("http") else domain_str
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "noscript", "iframe", "img", "svg"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)[:10000]
        state["kb_results"] = text
        
        log = {"agent": "Sourcing Agent", "action": f"Successfully scraped {len(text)} characters of context from {domain_str}.", "status": "success"}
        state["log_stream"].append(log)
    except Exception as e:
        error_msg = str(e)
        log = {"agent": "Sourcing Agent", "action": f"Web scraping failed: {error_msg}. Using generic context.", "status": "error"}
        state["log_stream"].append(log)
        state["kb_results"] = "Could not scrape the website content directly."
        
    state["current_agent"] = "copywriter_agent"
    state["messages"].append(AIMessage(content=f"Sourcing complete."))
    return state

async def copywriter_agent(state: AgentState):
    """Drafts the hyper-personalized cold email and json payload."""
    website_content = state["kb_results"]
    domain = state.get("company_domain", "the target company")
    
    log = {"agent": "Copywriter Agent", "action": "Synthesizing CRM payload and drafting personalized engagement...", "status": "running"}
    state["log_stream"].append(log)
    
    try:
        prompt = f"""You are a Principal Applied AI Architect named Farouk Hajjej. 
You build custom, production-ready autonomous agent swarms for enterprise clients. 
You are analyzing the website of {domain} to generate a hyper-personalized sales pitch.
Identify their core business and operational bottlenecks from the text below. If empty, guess. 
Pitch applied AI and autonomous agent swarms.

Website Content:
{website_content}

Output EXACTLY valid JSON with this schema:
{{
  "angle_detected": "string",
  "pain_targeted": "string",
  "subject_line": "string",
  "confidence_score": float,
  "cold_email_draft": "string - Format with paragraphs. Do not include 'Subject:' or greetings/signoffs."
}}
"""
        response = await llm.ainvoke(prompt)
        text = response.content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        try:
            data = json.loads(text)
            
            # Format the email draft nicely
            full_draft = f"Subject: {data.get('subject_line', '')}\n\nHi there,\n\n{data.get('cold_email_draft', '')}\n\nBest,\nFarouk"
            data["cold_email_draft"] = full_draft
            
            state["final_script"] = json.dumps(data, indent=2)
            
            log = {"agent": "Copywriter Agent", "action": "Sales execution payload generated. Ready for dispatch.", "status": "success"}
            state["log_stream"].append(log)
        except json.JSONDecodeError:
            log = {"agent": "Copywriter Agent", "action": "JSON Parsing failed. Outputting raw text.", "status": "error"}
            state["log_stream"].append(log)
            state["final_script"] = text
            
    except Exception as e:
        error_msg = str(e)
        log = {"agent": "Copywriter Agent", "action": f"Synthesis failed: {error_msg}", "status": "error"}
        state["log_stream"].append(log)
        state["final_script"] = json.dumps({"error": "Failed to generate payload"})
        
    state["current_agent"] = "end"
    state["messages"].append(AIMessage(content="Copywriting complete."))
    return state

# --- Routing ---
def route_next_agent(state: AgentState) -> str:
    return state["current_agent"]

# --- Build LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_agent)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("action_agent", action_agent)
workflow.add_node("sourcing_agent", sourcing_agent)
workflow.add_node("copywriter_agent", copywriter_agent)

workflow.set_entry_point("triage")
workflow.add_conditional_edges(
    "triage",
    route_next_agent,
    {
        "retrieval_agent": "retrieval_agent",
        "sourcing_agent": "sourcing_agent"
    }
)
workflow.add_conditional_edges(
    "retrieval_agent",
    route_next_agent,
    {"action_agent": "action_agent"}
)
workflow.add_conditional_edges(
    "sourcing_agent",
    route_next_agent,
    {"copywriter_agent": "copywriter_agent"}
)
workflow.add_edge("action_agent", END)
workflow.add_edge("copywriter_agent", END)

orchestrator = workflow.compile()

# --- API Endpoints ---

class TicketRequest(BaseModel):
    issue: str

@app.get("/")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "ok", "service": "Command Center API"}

class LeadCreate(BaseModel):
    email: str
    company_domain: str | None = None
    payload: Dict[str, Any] | None = None
    draft: str | None = None

class LeadUpdate(BaseModel):
    status: str

@app.post("/api/v1/leads")
async def create_lead(lead_in: LeadCreate, db: Session = Depends(get_db)):
    """Receives leads from the portfolio frontend and stores them in the SQLite DB."""
    # Try to extract company domain from payload if not provided directly
    company_domain = lead_in.company_domain
    if not company_domain and lead_in.payload:
        company_domain = lead_in.payload.get("company_domain", "")
        
    full_payload = json.dumps({
        "payload": lead_in.payload,
        "draft": lead_in.draft
    }) if lead_in.payload or lead_in.draft else None
    
    db_lead = Lead(
        email=lead_in.email,
        company_domain=company_domain,
        payload=full_payload
    )
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    
    lead_dict = {
        "id": db_lead.id,
        "email": db_lead.email,
        "company_domain": db_lead.company_domain,
        "status": getattr(db_lead, "status", "New"),
        "created_at": db_lead.created_at.isoformat() if db_lead.created_at else None,
        "payload": lead_in.payload if lead_in.payload else None,
        "draft": lead_in.draft if lead_in.draft else None
    }
    # Clean up the payload to just have one level if it was nested in create
    try:
        if db_lead.payload:
            payload_json = json.loads(db_lead.payload)
            lead_dict["payload"] = payload_json.get("payload", payload_json)
    except:
        pass

    asyncio.create_task(broadcast_lead(lead_dict))
    
    return {"status": "success", "lead_id": db_lead.id, "message": "Lead saved successfully"}

async def verify_admin(x_api_key: str = Header(None)):
    expected_key = os.getenv("ADMIN_PASSWORD")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not expected_key:
        print("WARNING: ADMIN_PASSWORD not set in environment. Access granted.")

@app.get("/api/v1/leads/stream")
async def stream_leads(api_key: str = None):
    """SSE endpoint for real-time lead updates."""
    expected_key = os.getenv("ADMIN_PASSWORD")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async def event_generator():
        q = asyncio.Queue()
        sse_clients.add(q)
        try:
            while True:
                lead_data = await q.get()
                yield {"event": "new_lead", "data": json.dumps(lead_data)}
        except asyncio.CancelledError:
            sse_clients.remove(q)
    
    return EventSourceResponse(event_generator())

@app.get("/api/v1/leads", dependencies=[Depends(verify_admin)])
async def get_leads(db: Session = Depends(get_db)):
    """Retrieves all leads from the database ordered by newest first."""
    leads = db.query(Lead).order_by(Lead.created_at.desc()).all()
    
    result = []
    for lead in leads:
        lead_dict = {
            "id": lead.id,
            "email": lead.email,
            "company_domain": lead.company_domain,
            "status": getattr(lead, "status", "New"),
            "created_at": lead.created_at.isoformat() if lead.created_at else None,
        }
        try:
            lead_dict["payload"] = json.loads(lead.payload) if lead.payload else None
        except:
            lead_dict["payload"] = lead.payload
            
        result.append(lead_dict)
        
    return {"status": "success", "leads": result}

@app.patch("/api/v1/leads/{lead_id}", dependencies=[Depends(verify_admin)])
async def update_lead_status(lead_id: int, lead_update: LeadUpdate, db: Session = Depends(get_db)):
    """Updates the status of a specific lead."""
    db_lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not db_lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    db_lead.status = lead_update.status
    db.commit()
    db.refresh(db_lead)
    return {"status": "success", "message": "Lead status updated"}

@app.post("/api/v1/orchestrate")
async def start_orchestration(request: TicketRequest):
    """
    In a real app, this might kick off a background task. 
    Here, we'll return a simple session ID or just process it.
    """
    initial_state = {
        "messages": [HumanMessage(content=request.issue)],
        "current_agent": "triage",
        "kb_results": "",
        "final_script": "",
        "log_stream": []
    }
    
    # Execute the graph (in a real app, this is where you'd stream)
    # Since we are mocking the stream for the portfolio UI, we'll 
    # handle the SSE streaming in a separate endpoint that manually 
    # yields events to look "agentic".
    return {"message": "Job started", "issue": request.issue}

@app.get("/api/v1/stream/{issue}")
async def stream_agent_execution(issue: str):
    """
    Server-Sent Events endpoint to stream the agent's 'thoughts' 
    to the Next.js frontend in real-time.
    """
    async def event_generator():
        # Setup state
        state = {
            "messages": [HumanMessage(content=issue)],
            "current_agent": "triage",
            "kb_results": "",
            "final_script": "",
            "log_stream": []
        }
        
        try:
            # We manually step through the compiled graph to yield progress
            # 1. Triage
            yield {"data": json.dumps({"agent": "System", "action": f"Received Request: '{issue}'", "status": "info"})}
            await asyncio.sleep(0.5)
            
            state = await triage_agent(state)
            for log in state["log_stream"]:
                yield {"data": json.dumps(log)}
                await asyncio.sleep(0.5)
            state["log_stream"] = [] # clear for next node
            
            if state["current_agent"] == "sourcing_agent":
                # SALES_LEAD Route
                yield {"data": json.dumps({"agent": "System", "action": "Routing to Sourcing Agent...", "status": "routing"})}
                await asyncio.sleep(0.5)
                
                state = await sourcing_agent(state)
                for log in state["log_stream"]:
                    yield {"data": json.dumps(log)}
                    await asyncio.sleep(0.5)
                state["log_stream"] = []

                yield {"data": json.dumps({"agent": "System", "action": "Routing to Copywriter Agent...", "status": "routing"})}
                await asyncio.sleep(0.5)
                
                state = await copywriter_agent(state)
                for log in state["log_stream"]:
                    yield {"data": json.dumps(log)}
                    await asyncio.sleep(0.5)
            else:
                # IT_SUPPORT Route
                yield {"data": json.dumps({"agent": "System", "action": "Routing to Retrieval Agent...", "status": "routing"})}
                await asyncio.sleep(0.5)
                
                state = await retrieval_agent(state)
                for log in state["log_stream"]:
                    yield {"data": json.dumps(log)}
                    await asyncio.sleep(0.5)
                state["log_stream"] = []
                
                # 3. Action
                yield {"data": json.dumps({"agent": "System", "action": "Routing to Action Agent...", "status": "routing"})}
                await asyncio.sleep(0.5)
                
                state = await action_agent(state)
                for log in state["log_stream"]:
                    yield {"data": json.dumps(log)}
                    await asyncio.sleep(0.5)
                
            # Final result payload
            final_payload = {
                "agent": "System", 
                "action": "Workflow Complete", 
                "status": "complete",
                "script": state["final_script"],
                "kb_context": state.get("kb_results", "")
            }
            yield {"data": json.dumps(final_payload)}
            
        except Exception as e:
            # Global catch-all to prevent infinite stream hangs on unexpected errors
            final_payload = {
                "agent": "System", 
                "action": f"Unrecoverable Orchestrator Error: {str(e)}", 
                "status": "error",
                "script": f"# CRITICAL ORCHESTRATOR ERROR\n# {str(e)}",
                "kb_context": state.get("kb_results", "")
            }
            yield {"data": json.dumps(final_payload)}
            
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
