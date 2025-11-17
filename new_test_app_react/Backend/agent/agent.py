"""
Business Insights LangGraph Agent with CopilotKit Integration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitSDK, LangGraphAgent
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# LangGraph imports
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

# CopilotKit imports
from copilotkit.langgraph import (
    copilotkit_customize_config,
    copilotkit_exit
)

# LLM imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Local imports
from .state import BusinessInsightsState
from .tools import (
    GENERATE_BUSINESS_INSIGHTS_TOOL,
    GENERATE_FORECAST_TOOL,
    GENERATE_VISUALIZATION_TOOL
)
from .prompts import BUSINESS_INSIGHTS_SYSTEM_PROMPT
from .flask_integration import flask_client
from .knowledge_base import (
    BUSINESS_UNITS_KNOWLEDGE,
    FORECASTING_MODELS_KNOWLEDGE,
    METRICS_INTERPRETATION,
    BUSINESS_INSIGHTS_TEMPLATES
)

load_dotenv()

# Configure OpenAI with NVIDIA API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

async def start_flow(state: Dict[str, Any], config: RunnableConfig):
    """Entry point for the agent flow"""

    # Initialize state fields if not present
    if "business_units" not in state:
        state["business_units"] = []
    if "forecast_data" not in state:
        state["forecast_data"] = []
    if "analysis_results" not in state:
        state["analysis_results"] = {}
    if "visualizations" not in state:
        state["visualizations"] = []

    return Command(
        goto="chat_node",
        update={
            "messages": state["messages"],
            "business_units": state["business_units"],
            "forecast_data": state["forecast_data"],
            "analysis_results": state["analysis_results"],
            "visualizations": state["visualizations"],
        }
    )

async def chat_node(state: Dict[str, Any], config: RunnableConfig):
    """Main chat node for processing user queries"""

    # Initialize the LLM with NVIDIA API
    model = ChatOpenAI(
        model="meta/llama-3.1-70b-instruct",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.7,
    )

    # Configure for streaming
    if config is None:
        config = RunnableConfig(recursion_limit=25)

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[
            {"state_key": "business_units", "tool": "generate_business_insights"},
            {"state_key": "forecast_data", "tool": "generate_forecast"},
            {"state_key": "visualizations", "tool": "generate_visualization"},
        ],
    )

    # Bind tools to model
    model_with_tools = model.bind_tools(
        [
            *state["copilotkit"]["actions"],
            GENERATE_BUSINESS_INSIGHTS_TOOL,
            GENERATE_FORECAST_TOOL,
            GENERATE_VISUALIZATION_TOOL,
        ],
        parallel_tool_calls=False,
    )

    # Enhanced system prompt with knowledge
    ENHANCED_SYSTEM_PROMPT = f"""
    {BUSINESS_INSIGHTS_SYSTEM_PROMPT}

    KNOWLEDGE BASE:

    {BUSINESS_UNITS_KNOWLEDGE}

    {FORECASTING_MODELS_KNOWLEDGE}

    {METRICS_INTERPRETATION}

    {BUSINESS_INSIGHTS_TEMPLATES}

    Use this knowledge to provide accurate, contextual responses.
    """

    # Generate response
    response = await model_with_tools.ainvoke([
        SystemMessage(content=ENHANCED_SYSTEM_PROMPT),
        *state["messages"],
    ], config)

    # Update messages
    messages = state["messages"] + [response]

    # Handle tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call.name if hasattr(tool_call, "name") else tool_call.get("name")
        tool_args = tool_call.args if hasattr(tool_call, "args") else tool_call.get("args", {})

        # Process different tool calls
        if tool_name == "generate_business_insights":
            state["analysis_results"] = tool_args
        elif tool_name == "generate_forecast":
            forecast_result = await flask_client.generate_forecast(
                weeks=tool_args.get("weeks"),
                model_type=tool_args.get("model_type"),
            )
            state["forecast_data"] = forecast_result
        elif tool_name == "generate_visualization":
            state["visualizations"].append(tool_args)

    await copilotkit_exit(config)
    return Command(
        goto=END,
        update={
            "messages": messages,
            "business_units": state["business_units"],
            "forecast_data": state["forecast_data"],
            "analysis_results": state["analysis_results"],
            "visualizations": state["visualizations"],
        }
    )

# Build the graph
workflow = StateGraph(BusinessInsightsState)
workflow.add_node("start_flow", start_flow)
workflow.add_node("chat_node", chat_node)
workflow.set_entry_point("start_flow")
workflow.add_edge(START, "start_flow")
workflow.add_edge("start_flow", "chat_node")
workflow.add_edge("chat_node", END)

# Compile
business_graph = workflow.compile(checkpointer=MemorySaver())

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sdk = CopilotKitSDK(
    agents=[
        LangGraphAgent(
            name="business_insights_agent",
            description="AI agent for business intelligence and forecasting",
            graph=business_graph,
        )
    ]
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server"""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "agent.agent:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )

if __name__ == "__main__":
    main()
