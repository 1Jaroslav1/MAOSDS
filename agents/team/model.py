from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Sequence, Annotated, List
from agents.hub import gpt_4o_mini, get_tavily_tool
from langgraph.graph import StateGraph, START


class TeamMember(TypedDict):
    name: str
    description: str


class TeamState(TypedDict):
    topic: str
    opposite_team_arguments: List[BaseMessage]
    members: List[TeamMember]
    team_leader_advice: str
    executed_members: List[TeamMember]
    team_arguments: List[str]
