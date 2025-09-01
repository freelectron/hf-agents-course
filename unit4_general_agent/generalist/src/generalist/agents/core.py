from typing import Type
from dataclasses import dataclass

class BaseAgentCapability:
    """Base class for all agent capabilities."""
    name: str  # Defines that all subclasses should have a 'name' class attribute

    def __init__(self, activity: str = None):
        """
        Initializes the capability with a specific activity.

        Args:
            activity: The specific action to be performed.
        """
        self.activity = activity

    def __repr__(self) -> str:
        """Provides a clean string representation of the object."""
        # 'self.name' correctly accesses the class attribute from the instance
        return f"{self.__class__.__name__}(name='{self.name}', activity='{self.activity}')"

class AgentCapabilityDeepWebSearch(BaseAgentCapability):
    """Capability for performing a deep web search."""
    name = "deep_web_search"

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

class AgentCapabilityCodeMathWritter(BaseAgentCapability):
    """Capability for writing code or solving math problems."""
    name = "code_math_writing"

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
    AgentCapabilityCodeMathWritter.name: AgentCapabilityCodeMathWritter,
}

def json_to_capability_plan(json_data: dict) -> CapabilityPlan:
    """
    Convert JSON response into a CapabilityPlan with proper capability objects.
    """
    subplan = []
    for step in json_data["subplan"]:
        cap_name = step["capability"]
        activity = step["activity"]

        if cap_name not in CAPABILITY_MAP:
            raise ValueError(f"Unknown capability: {cap_name}")

        cap_class = CAPABILITY_MAP[cap_name]
        capability = cap_class(activity=activity)
        subplan.append(capability)

    return CapabilityPlan(subplan=subplan)