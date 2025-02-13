from team_member import create_team_member_workflow
from state import TeamMemberState, TeamRole

team_member_1 = create_team_member_workflow()

team_member_state: TeamMemberState = {
    "topic": "Should artificial intelligence have legal rights and responsibilities, similar to humans and corporations?",
    "team_role": TeamRole.PROPOSING,
    "person": {
        "name": "Dr. Alex Carter",
        "expertise": "AI Ethics and Law",
        "description": "A leading expert in artificial intelligence regulations, focusing on the intersection of ethics, law, and AI development."
    },
    "transcript": [],
    "team_arguments": [],
    "opponent_arguments": [],
    "audience_profile": {
        "demographics": "Academics, policymakers, and technology professionals",
        "interests": ["AI ethics", "law", "technological impact", "human rights"]
    },
    "analysis": {},
    "retrieved_data": {},
    "argument": {},
    "lexicon_adjustment": {},
    "evaluation": {},
    "iteration_number": 0
}

team_member_graph_1 = team_member_1.compile()
team_member_graph_1.get_graph().draw_mermaid_png(output_file_path ="team_member_graph_1.png")


events = team_member_graph_1.stream(team_member_state, stream_mode="values")

for event in events:
    print(event)
