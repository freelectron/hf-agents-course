from typing import Optional, Type
from dataclasses import dataclass

from ..tools.summarisers import construct_short_answer
from ..tools.text_processing import text_process_llm
from ..tools.web_search import web_search 
from ..tools.data_model import ContentResource, ShortAnswer
from ..tools.code import write_python_eda, run_code, write_python_code_task


@dataclass
class AgentCapabilityOutput: 
    # Based on resources or file attachments, a list of short answers to the task 
    answers: Optional[list[ShortAnswer]]
    # Produced resources
    resources: Optional[list[ContentResource]] = None


class BaseAgentCapability:
    """Base class for all agent capabilities."""
    name: str  # Defines that all subclasses should have a 'name' class attribute

    def __init__(self, activity: str):
        """
        Initializes the capability with a specific activity.

        Args:
            activity (str): The specific action to be performed.
        """
        # TODO: not sure how good of an idea to init an agent with an activity in the state
        self.activity = activity

    def __repr__(self) -> str:
        """Provides a clean string representation of the object."""
        # 'self.name' correctly accesses the class attribute from the instance
        return f"{self.__class__.__name__}(name='{self.name}', activity='{self.activity}')"
    
    def run(*args, **kwargs) -> AgentCapabilityOutput:
        """
        Execute the main logic of the agent.
        """
        raise NotImplementedError("Method `run` is not implemented")


class AgentCapabilityDeepWebSearch(BaseAgentCapability):
    """Capability for performing a deep web search."""
    name = "deep_web_search"

    def run(self) -> AgentCapabilityOutput:
        """
        Args:
         ask (str): what we are searching
        """
        return AgentCapabilityOutput(resources=web_search(self.activity))


class AgentCapabilityVideoProcessor(BaseAgentCapability):
    """Capability for processing video files."""
    name = "video_processing"


class AgentCapabilityAudioProcessor(BaseAgentCapability):
    """Capability for processing audio files."""
    name = "audio_processing"


class AgentCapabilityImageProcessor(BaseAgentCapability):
    """Capability for processing image files."""
    name = "image_processing"


class AgentCapabilityStructuredDataProcessor(BaseAgentCapability):
    """Capability for processing structured data."""
    name = "structured_data_processing"


class AgentCapabilityUnstructuredDataProcessor(BaseAgentCapability):
    """Capability for processing unstructured text."""
    name = "unstructured_data_processing"

    def run(self, resources: list[ContentResource]) -> AgentCapabilityOutput:
        """
        Args:
        ask (str): what data we are analysing 
        """
        resource_contents = [web_resource.content for web_resource in resources]

        # TODO: make this more robust, also now can only handle resources that have text content 
        text  = "; \n\n |  ".join(resource_contents)

        answers = text_process_llm(self.activity, text)
        short_answers = [construct_short_answer(self.activity, answer) for answer in answers] 

        return AgentCapabilityOutput(
            answers=short_answers
        )


class AgentCapabilityCodeWritterExecutor(BaseAgentCapability):
    """Capability for writing and executing python code"""
    name = "code_writing_execution"

    def run(self, resources:list[ContentResource]) -> AgentCapabilityOutput:
        """ 
        """
        # Analyse the given resources, determine what the files contain (EDA)
        eda_code = write_python_eda(resources)
        eda_results = run_code(eda_code)        
        
        # Given the activity=task and what EDA results determine the final code that would produce the result
        task_code = write_python_code_task(task=self.activity, eda_results=eda_results, resources=resources)
        results = run_code(task_code)
        short_answers = [construct_short_answer(self.activity, results)] 
        
        return AgentCapabilityOutput(
            answers=short_answers
        ) 

@dataclass
class CapabilityPlan:
    """A structured plan outlining the sequence of capabilities and actions."""
    subplan: list[BaseAgentCapability]


CAPABILITY_MAP: dict[str, Type[BaseAgentCapability]] = {
    AgentCapabilityDeepWebSearch.name: AgentCapabilityDeepWebSearch,
    AgentCapabilityVideoProcessor.name: AgentCapabilityVideoProcessor,
    AgentCapabilityAudioProcessor.name: AgentCapabilityAudioProcessor,
    AgentCapabilityImageProcessor.name: AgentCapabilityImageProcessor,
    AgentCapabilityStructuredDataProcessor.name: AgentCapabilityStructuredDataProcessor,
    AgentCapabilityUnstructuredDataProcessor.name: AgentCapabilityUnstructuredDataProcessor,
    AgentCapabilityCodeWritterExecutor.name: AgentCapabilityCodeWritterExecutor,
}

def json_to_capability_plan(json_data: dict, capability_map: dict[str, Type[BaseAgentCapability]] = CAPABILITY_MAP) -> CapabilityPlan:
    """
    Convert JSON response into a CapabilityPlan with proper capability objects.
    """
    subplan = []
    for step in json_data["subplan"]:
        cap_name = step["capability"]
        activity = step["activity"]

        if cap_name not in capability_map:
            raise ValueError(f"Unknown capability: {cap_name}")

        cap_class = capability_map[cap_name]
        capability = cap_class(activity=activity)
        subplan.append(capability)

    return CapabilityPlan(subplan=subplan)