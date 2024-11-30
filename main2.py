from agents import TeamMember, TeamState, create_team_graph
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

team_members = [
    TeamMember(name="Alice", description="AI ethics expert"),
    TeamMember(name="Bob", description="Data scientist specializing in NLP"),
    TeamMember(name="Charlie", description="AI governance researcher"),
]

builder = create_team_graph("New technologies, such as AI and quantum computing pose a risk to open, just, democratic societies and could easily undermine them", members=team_members)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path ="output_graph.png")

state: TeamState = {
    "topic": "New technologies, such as AI and quantum computing pose a risk to open, just, democratic societies and could easily undermine them",
    "opposite_team_arguments": [],
    "members": team_members,
    "team_leader_advice": "",
    "executed_members": [],
    "team_arguments": [],
}

events = graph.stream(state, stream_mode="values")
for event in events:
    print(event)
