from agents import TeamMember, create_team_graph, DebateState, chairman_node, Team
from typing_extensions import Dict
from langgraph.graph import StateGraph, START


pros_team_members = [
    TeamMember(name="Alice", description="AI ethics expert"),
    TeamMember(name="Bob", description="Data scientist specializing in NLP"),
    TeamMember(name="Charlie", description="AI governance researcher"),
]

pros_team_builder = create_team_graph(members=pros_team_members)
pros_team = pros_team_builder.compile()


cons_team_members = [
    TeamMember(name="Diana", description="Ethical researcher concerned about AI's societal impact"),
    TeamMember(name="Ethan", description="Economist analyzing AI-driven economic disruptions"),
    TeamMember(name="Fiona", description="Privacy advocate emphasizing risks of AI surveillance"),
]

cons_team_builder = create_team_graph(members=cons_team_members)
cons_team = cons_team_builder.compile()


def call_pros_team(state: DebateState):
    response = pros_team.invoke(state["next_team_state"])
    team_arguments: Dict = state["team_arguments"]
    team_arguments[state["next"]] = response["team_arguments"]
    return {
        "team_arguments": team_arguments
    }


def call_cons_team(state: DebateState):
    response = cons_team.invoke(state["next_team_state"])
    team_arguments: Dict = state["team_arguments"]
    team_arguments[state["next"]] = response["team_arguments"]
    return {
        "team_arguments": team_arguments
    }


teams = [
    Team(name="pros_team", topic="New technologies, such as AI and quantum computing, have great potential and will enhance open, just, and, democratic societies", members=pros_team_members),
    Team(name="cons_team", topic="New technologies, such as AI and quantum computing pose a risk to open, just, democratic societies and could easily undermine them", members=cons_team_members)
]

workflow = StateGraph(DebateState)
workflow.add_node("chairman_node", chairman_node)

workflow.add_node("pros_team", call_pros_team)
workflow.add_node("cons_team", call_cons_team)

workflow.add_edge(START, "chairman_node")
workflow.add_conditional_edges("chairman_node", lambda state: state["next"])
workflow.add_edge("pros_team", "chairman_node")
workflow.add_edge("cons_team", "chairman_node")


debate = workflow.compile()
debate.get_graph().draw_mermaid_png(output_file_path ="output_graph.png")
state = {
    "teams": teams,
    "team_arguments": {},
    "executed_teams": [],
    "next_team_state": {},
    "next": ""
}
events = debate.stream(state, stream_mode="values")
for event in events:
    print(event)
