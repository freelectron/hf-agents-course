import json

from ..agents.core import AgentCapabilityAudioProcessor, AgentCapabilityCodeMathWritter, AgentCapabilityDeepWebSearch, AgentCapabilityImageProcessor, AgentCapabilityStructuredDataProcessor, AgentCapabilityUnstructuredDataProcessor, AgentCapabilityVideoProcessor, CapabilityPlan, json_to_capability_plan
from .data_model import Task
from ..models.core import llm

def create_plan(task: str) -> str:
    """
    Given a task, determine a step-by-step action plan of what needs to be done to accomplish this task and output the answer/result. 
    The most important actions that are taken: 
     1. Define the goal: what result is asked to be produced.
     2. List the steps: provide a short explanation for each action that needs to be taken.       
    """
    
    prompt = f"""
You are an expert project planner. Your task is to create a concise, step-by-step action plan to accomplish the user's goal.

User's Goal:
---
{task}
---

Instructions:
1. Clarify the Core Objective: Start by rephrasing the user's goal as a single, clear, and specific objective.
2. Develop a Chronological Action Plan: Break down the objective into a logical sequence of high-level steps.

Guiding Principles for the Plan:
- Tool-Agnostic: Focus on the action required, not the specific tool to perform it (e.g., use "Gather data on market trends" instead of "Search Google for market trends").
- Information First: The initial step should almost always be to gather and analyze the necessary information before taking further action.
- S.M.A.R. Steps: Each step must be Specific, Measurable, Achievable, and Relevant. The focus is on the logical sequence, not specific deadlines.
- Concise: Include only the critical steps needed to reach the objective.

Example Output Format (ALWAS **JSON** ):
{{
  "objective": "Plan and execute a one-day offsite event for a team of 10 people focused on team building and strategic planning.",
  "plan": [
    "Gather requirements including budget, potential dates, and key goals for the offsite from team leadership",
    "Research and shortlist suitable venues and activity options that fit the budget and goals",
    "Create a detailed agenda and budget proposal for approval",
    "Book the selected venue, catering, and activities upon approval",
    "Send out official invitations and manage attendee confirmations and dietary requirements",
    "Finalize all logistical details and communicate the full itinerary to the team"
  ]
}}
where
  "objective" 's value in the json is a clear, one-sentence summary of the end goal,
  "plan" 's value in the json is a list **ALWAYS SEPARATED BY PYTHON NEWLINE CHARCTER** like 
  [
    A short explanation of the first logical step", 
    A short explanation of the next step that follows from the first",
    And so on..."
  ]
  **IMPORTANT**: do not include any json formating directives, output plain json string
"""
    task_response = llm.complete(prompt)

    return task_response.text

def determine_capabilities(task: str, attachments: list[str] = None) -> CapabilityPlan:
    """
    Analyzes a task and generates a sequential execution plan using available capabilities.

    Args:
        task (str): The description of the task to be performed.
        attachments (list[str]): A list of file names related to the task.

    Returns:
        CapabilityPlan: A dataclass containing the ordered list of sub-tasks.
    """
    attachment_info = ""
    if attachments:
        attachment_info = f"\n\nAttachments provided: {', '.join(attachments)}"

    # TODO: is this fine to map capability to an agent one-to-one? 
    planning_prompt = f"""
You are a highly intelligent planning agent. Your primary function is to analyze a user's task and create a precise, step-by-step execution plan using a predefined set of capabilities.

**Your Task:**
Analyze the provided task and create a sequential plan to accomplish it. The plan should be a list of steps, where each step defines the capability to use and the specific activity to perform.

**Capabilities:**
- `{AgentCapabilityDeepWebSearch.name}`: Find, evaluate, and download web content (e.g., articles, documents). This capability is for search and downloading web resources only, not for processing the content or getting any answers on the content.
- `{AgentCapabilityVideoProcessor.name}`: Download video, extract frames or audio from a video file for further analysis.
- `{AgentCapabilityAudioProcessor.name}`: Download audio, transcribe speech, identify sounds, or analyze properties of an audio file.
- `{AgentCapabilityImageProcessor.name}`: Download image, analyze an image to identify objects, read text (OCR), or understand its content.
- `{AgentCapabilityStructuredDataProcessor.name}`: Analyze, query, or visualize data from structured files like Parquet, CSV, JSON, or databases.
- `{AgentCapabilityUnstructuredDataProcessor.name}`: Analyze, summarize, extract information from, or answer questions about raw text or documents (e.g., PDFs, TXT files, retrieved web content).
- `{AgentCapabilityCodeMathWritter.name}`: Generate or execute code, solve mathematical problems, or perform complex logical operations and computations.

Instructions:
Deconstruct the Task -> Assign Capabilities for each step -> Define the Activity for each step (i.e.,write a clear and concise description of the specific action to be performed using the chosen capability)

Example 1: Simple Fact Lookup
Task: "What is the boiling point of water at sea level?"
Output:
{{
  "subplan": [
    {{
      "capability": "{AgentCapabilityDeepWebSearch.name}",
      "activity": "Search for the boiling point of water at sea level"
    }},
    {{
      "capability": "{AgentCapabilityUnstructuredDataProcessor.name}",
      "activity": "Analyze the downloaded web resources and find the reference to the boiling point temperature."
    }}
  ]
}}

Example 2: Multi-step Information Retrieval and Analysis

Task: "Find the Q2 2025 earnings report for NVIDIA and tell me what their 'Gaming' division revenue was."
Output:
{{
  "subplan": [
    {{
      "capability": "{AgentCapabilityDeepWebSearch.name}",
      "activity": "Search for and download NVIDIA's official Q2 2025 earnings report document and download it."
    }},
    {{
      "capability": "{AgentCapabilityUnstructuredDataProcessor.name}",
      "activity": "Analyze the downloaded earnings report to find and extract the revenue figure for the 'Gaming' division."
    }}
  ]
}}

---
Begin Plan Generation

Task: "{task}"
Attachments: "{attachment_info}"

Respond in this exact JSON format:
{{
  "subplan": [
    {{
      "capability": "...",
      "activity": "..."
    }},
    {{
      "capability": "...",
      "activity": "..."
    }}
  ]
}}
"""
    response = llm.complete(planning_prompt)
    response_text = response.text.strip()

    result = json.loads(response_text)

    return json_to_capability_plan(result)