import json

from ..agents.core import AgentCapabilityAudioProcessor, AgentCapabilityCodeWritterExecutor, AgentCapabilityDeepWebSearch, AgentCapabilityImageProcessor, AgentCapabilityStructuredDataProcessor, AgentCapabilityUnstructuredDataProcessor, AgentCapabilityVideoProcessor, CapabilityPlan, json_to_capability_plan
from .data_model import Task
from ..models.core import llm
from ..tools.data_model import ContentResource

from generalist import logger

def create_plan(task: str, resources:list[ContentResource]) -> str:
    """
    Given a task, determine a step-by-step action plan of what needs to be done to accomplish this task and output the answer/result. 
    The most important actions that are taken: 
     1. Define the goal: what result is asked to be produced.
     2. List the steps: provide a short explanation for each action that needs to be taken.       
    """
    
    prompt = f"""
You are an expert project planner. Your task is to create a concise, step-by-step action plan to accomplish the given user's goal and available resources.

User's Goal:
{task}
---
Available resources:
{resources}
---

Instructions:
1. Clarify the Core Objective: Start by rephrasing the user's goal as a single, clear, and specific objective.
2. Develop a Chronological Action Plan: Break down the objective into a logical sequence of high-level steps (aim for 1 or 2 steps).

Example Output Format (ALWAS **JSON** ):
{{
  "objective": "Produce a plot with the sales data of the provided csv (/home/user_name/datat/company_balance_sheet.csv)",
  "plan": [
    "Analyse the information on what columns are present in the csv and which ones represent sales",
    "Produce a piece of code that would only plot sales data"
  ]
}}
where
  "objective" 's value in the json is a clear, one-sentence summary of the end goal,
  "plan" 's value in the json is a list **ALWAYS SEPARATED BY PYTHON NEWLINE CHARCTER** like 
  [
    "A short explanation of the first logical step", 
    "Thjeconcluding step",
  ]
  **IMPORTANT**: you should only include the minimum number of steps to accomplish the task (*strive for 1 or 2*), do not include varification steps.
  **IMPORTANT**: do not include any json formating directives, output plain json string.
"""    
    task_response = llm.complete(prompt)

    return task_response.text


def determine_capabilities(current_step: str, task: Task, resources: list[ContentResource] = list(), context: str = "") -> CapabilityPlan:
    """
    Analyzes a task and generates a sequential execution plan using available capabilities.
    
    TODO: implement
      - `{AgentCapabilityStructuredDataProcessor.name}`: Analyze, query, or visualize data from structured files like Parquet, CSV, JSON, or databases.
      - `{AgentCapabilityImageProcessor.name}`: Download image, analyze an image to identify objects, read text (OCR), or understand its content.
      - `{AgentCapabilityVideoProcessor.name}`: Download video, extract frames or audio from a video file for further analysis.
      - `{AgentCapabilityAudioProcessor.name}`: Download audio, transcribe speech, identify sounds, or analyze properties of an audio file.

    Args:
        task (str): The description of the task to be performed.
        attachments (list[str]): A list of file names related to the task.

    Returns:
        CapabilityPlan: A dataclass containing the ordered list of sub-tasks.
    """
    full_context = ""
    if resources:
        full_context = f"\nAttached resources: {resources}"

    # TODO: is this fine to map capability to an agent one-to-one? 
    planning_prompt = f"""
You are a highly intelligent planning agent. Your primary function is to analyze a user's task together with the current step that needs to be executed and given the context/attachments create a precise, step-by-step plan using only a predefined set of capabilities.

Capabilities:
- `{AgentCapabilityDeepWebSearch.name}`: Find, evaluate, and download web content (e.g., articles, documents). This capability is for search and downloading web resources only, not for processing the content or getting any answers on the content.
- `{AgentCapabilityUnstructuredDataProcessor.name}`: Analyze, summarize, extract information from, or answer questions about raw text or documents (e.g., PDFs, TXT files, retrieved web content).
- `{AgentCapabilityCodeWritterExecutor.name}`: Generate or execute code, solve mathematical problems, or perform complex logical operations and computations on files.

Your Task:
Analyze the provided task and create a sequential plan to accomplish it. The plan should be a list of steps, where each step defines the capability to use and the specific activity to perform, thus:
- "activity" for the step is a clear and concise description of the specific action to be performed using the chosen capability
- "capablity" is one of the above mentioned capabilities that should be used to accomplish the activity

Example 1:
Current step: "Look up the age of the that actor." 
Task: "Task(question='What is the age of the main actor of Inception?', objective='Identify the main actor who played in Inception and their age', plan=['Determine the main charcter of the movie Inception', "Look up the age of the that actor'"]),
Context: {{'found': 'Leonardo DiCaprio played the main character in Inception' ...}} 
Output:
{{
  "subplan": [
    {{
      "activity": "Search for the age of Leonardo DiCaprio online",
      "capability": "{AgentCapabilityDeepWebSearch.name}"
    }},
    {{
      "activity": "Identify the age of Leonardo DiCaprio from the text",
      "capability": "{AgentCapabilityUnstructuredDataProcessor.name}"
    }}
  ]
}}

Example 2: Audio Content Extraction
Current step: "Extract the spoken content from the audio file."
Task: "Task(question='What are the main topics discussed in the uploaded podcast episode?', objective='Summarize the key themes of the podcast', plan=['Download the audio file', 'Extract and transcribe speech', 'Summarize the transcription'])",
Context: {{'file': 'podcast_episode.mp3'}}
Output:
{{
    "subplan": [
  {{
    "activity": "Write code to download and transcribe the speech from podcast_episode.mp3",
    "capability": "{AgentCapabilityCodeWritterExecutor.name}"
  }}
  ]
}}
---
Current step: "{current_step}".
Task: "{task}".
Context: 
"{context}"
"{full_context}"

**IMPORTANT**: you are only handling "{current_step}" of the plan "{task.plan}", FOCUS ONLY ON THAT AND WHAT WAS ALREADY FOUND IN THE CONTEXT.   
**IMPORTANT**: take into account the context and attachments, e.g., specify the activity based on what was already FOUND in the context.     
ONLY RESPOND WITH A SINGLE JSON, in this exact JSON format:
{{
  "subplan": [
    {{
      "activity": "...",
      "capability": "..."
    }},
    {{
      "activity": "...",
      "capability": "..."
    }}
  ]
}}
"""   
    response = llm.complete(planning_prompt)
    response_text = response.text.strip()

    result = json.loads(response_text)

    return json_to_capability_plan(result)