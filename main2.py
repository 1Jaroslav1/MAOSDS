from agents import TeamMember, TeamState, create_team_graph
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Sequence, Annotated, List, Dict
from agents.hub import gpt_4o_mini, get_tavily_tool
from langgraph.graph import StateGraph, START

load_dotenv(find_dotenv())

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

#
# graph.get_graph().draw_mermaid_png(output_file_path ="output_graph.png")
#
# state: TeamState = {
#     "topic": "New technologies, such as AI and quantum computing pose a risk to open, just, democratic societies and could easily undermine them",
#     "opposite_team_arguments": [],
#     "members": team_members,
#     "team_leader_advice": "",
#     "executed_members": [],
#     "team_arguments": [],
# }
#
# events = graph.stream(state, stream_mode="values")
# for event in events:
#     print(event)


class Team(TypedDict):
    name: str
    topic: str
    members: List[TeamMember]


class DebateState(TypedDict):
    teams: List[Team]
    team_arguments: Dict[str, List[str]]
    executed_teams: List[Team]
    next_team_state: TeamState
    next: str


class ChairmanOutput(BaseModel):
    next_team: str = Field(description="Next team")

def chairman_node(state: DebateState):
    teams = state["teams"]
    executed_teams = state["executed_teams"]
    remaining_teams = [t for t in teams if t not in executed_teams]

    if remaining_teams:
        chairman_prompt = PromptTemplate(
            template="""
                    You are the chairman overseeing an ongoing debate among multiple teams. Your role is to ensure each team presents its arguments in a cohesive and strategic manner, guiding the overall debate flow. Below is the current status of the debate:

                    **Current Debate Status:**
                    - **Teams Participating:** {teams}.
                    - **Teams That Have Already Presented:** {executed_teams}.
                    - **Remaining Teams:** {remaining_teams}.

                    **Your Responsibilities:**
                    1. **Select the Next Team:** Randomly choose team ({remaining_teams}) to present next.
                """,
            input_variables=["teams", "executed_teams", "remaining_teams", "team_arguments"]
        )

        llm_chain = chairman_prompt | gpt_4o_mini.with_structured_output(ChairmanOutput)
        result = llm_chain.invoke(
            {
                "teams": teams,
                "executed_teams": executed_teams,
                "remaining_teams": remaining_teams,
            }
        )

        next_team_name = result.next_team
        opposite_team_arguments = [
            arg for team, args in state["team_arguments"].items() if team != next_team_name for arg in args
        ]
        next_team = None
        for team in teams:
            if team["name"] == next_team_name:
                next_team = team

        next_team_state = {
            "topic": next_team["topic"],
            "opposite_team_arguments": opposite_team_arguments,
            "members": next_team["members"],
            "team_leader_advice": "",
            "executed_members": [],
            "team_arguments": []
        }

        return {
            "executed_teams": state["executed_teams"] + [next_team],
            "next": next_team_name,
            "next_team_state": next_team_state
        }
    else:
        return {
            "next": END
        }


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
