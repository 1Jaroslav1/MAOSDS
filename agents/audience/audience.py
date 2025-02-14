from state import AudienceMember, AudienceState
from typing_extensions import TypedDict, Sequence, Annotated, List
from member import create_member
from langgraph.graph import StateGraph, START, END


def create_audience(members: List[AudienceMember]):
    workflow = StateGraph(AudienceState)

    for member in members:
        workflow.add_node(member["name"], create_member("init", member))
        workflow.add_edge(START, member["name"])
        workflow.add_edge(member["name"], END)

    return workflow
