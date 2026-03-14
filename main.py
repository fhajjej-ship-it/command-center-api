import os
import json
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from dotenv import load_dotenv

from mocks import search_knowledge_base

load_dotenv()

app = FastAPI(title="Command Center API", version="1.0.0")

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

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Nodes (The Agents) ---

async def triage_agent(state: AgentState):
    """Analyzes the initial request and decides next steps."""
    user_request = state["messages"][0].content
    
    log = {"agent": "Triage AI", "action": "Analyzing incoming IT support ticket...", "status": "running"}
    state["log_stream"].append(log)
    
    try:
        prompt = f"Analyze this IT support ticket to determine if it requires knowledge retrieval. Ticket: '{user_request}'. Respond with a short, 1-sentence summary of the intent."
        response = await llm.ainvoke(prompt)
        intent = response.content.strip().replace("\n", " ")
        
        log = {"agent": "Triage AI", "action": f"Identified intent: '{intent[:100]}...'. Routing to Retrieval Agent.", "status": "success"}
        state["log_stream"].append(log)
        
        state["current_agent"] = "retrieval_agent"
        state["messages"].append(AIMessage(content=f"Triage complete. Intent: {intent}"))
    except Exception as e:
        error_msg = str(e)
        log = {"agent": "Triage AI", "action": f"Error analyzing ticket: {error_msg[:100]}... Routing to Fallback.", "status": "error"}
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

# --- Routing ---
def route_next_agent(state: AgentState) -> str:
    return state["current_agent"]

# --- Build LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_agent)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("action_agent", action_agent)

workflow.set_entry_point("triage")
workflow.add_conditional_edges(
    "triage",
    route_next_agent,
    {"retrieval_agent": "retrieval_agent"}
)
workflow.add_conditional_edges(
    "retrieval_agent",
    route_next_agent,
    {"action_agent": "action_agent"}
)
workflow.add_edge("action_agent", END)

orchestrator = workflow.compile()

# --- API Endpoints ---

class TicketRequest(BaseModel):
    issue: str

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
            yield {"data": json.dumps({"agent": "System", "action": f"Received IT Ticket: '{issue}'", "status": "info"})}
            await asyncio.sleep(1)
            
            state = await triage_agent(state)
            for log in state["log_stream"]:
                yield {"data": json.dumps(log)}
                await asyncio.sleep(0.5)
            state["log_stream"] = [] # clear for next node
                
            # 2. Retrieval
            yield {"data": json.dumps({"agent": "System", "action": "Routing to Retrieval Agent...", "status": "routing"})}
            await asyncio.sleep(1)
            
            state = await retrieval_agent(state)
            for log in state["log_stream"]:
                yield {"data": json.dumps(log)}
                await asyncio.sleep(0.5)
            state["log_stream"] = []
            
            # 3. Action
            yield {"data": json.dumps({"agent": "System", "action": "Routing to Action Agent...", "status": "routing"})}
            await asyncio.sleep(1)
            
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
                "kb_context": state["kb_results"]
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
