from state import TeamState, TeamRole
from team import create_team_workflow

members = [
        {
            "name": "Alice",
            "expertise": "Artificial Intelligence",
            "description": "An AI specialist with deep expertise in machine learning, neural networks, and natural language processing."
        },
        {
            "name": "Bob",
            "expertise": "Conservation",
            "description": "A seasoned conservator who values traditional methods and proven techniques, with decades of experience."
        },
        {
            "name": "Charlie",
            "expertise": "Economics",
            "description": "A dynamic young economics student with fresh insights on market trends and innovative financial analysis."
        }
    ]

team_state: TeamState = {
    "topic": "Should artificial intelligence have legal rights and responsibilities, similar to humans and corporations?",
    "team_role": TeamRole.PROPOSING,
    "members": members,
    "transcript": [],
    "audience_profile": {
        "demographics": "Academics, policymakers, and technology professionals",
        "interests": ["AI ethics", "law", "technological impact", "human rights"]
    }
}

team_workflow = create_team_workflow(members)
team = team_workflow.compile()

team.get_graph().draw_mermaid_png(output_file_path ="team.png")

events = team.stream(team_state, stream_mode="values")

for event in events:
    print(event)
