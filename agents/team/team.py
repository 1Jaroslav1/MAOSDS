from langgraph.prebuilt import ToolNode
from typing_extensions import List
from agents.hub import get_tavily_tool
from langgraph.graph import StateGraph, START
from .model import TeamState, TeamMember
from .team_leader import team_leader_node
from .team_member import TeamMemberNode

tavily_tool = get_tavily_tool(max_results=5)
tools = [tavily_tool]


def is_all_members_executed(state: TeamState):
    if len(state["executed_members"]) == len(state["members"]):
        return


def create_team_graph(members: List[TeamMember]) -> StateGraph:
    team_tools_node = ToolNode(tools)

    workflow = StateGraph(TeamState)
    workflow.add_node("team_leader_node", team_leader_node)
    workflow.add_node("team_tools_node", team_tools_node)
    
    for member in members:
        team_member_node = TeamMemberNode(member=member)

        member_node = team_member_node.create_member_node()
        member_name = member["name"]

        workflow.add_node(member_name, member_node)
        workflow.add_edge(member_name, "team_leader_node")

    workflow.add_edge(START, "team_leader_node")
    workflow.add_conditional_edges("team_leader_node", lambda state: state["next"])

    return workflow
