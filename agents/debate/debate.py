import random
from agents.debate.state import DebateState
from langgraph.graph import StateGraph, START, END
from agents.audience.audience import create_audience
from agents.team import create_team_workflow
from agents.model.model import TeamRole
from typing_extensions import List
from agents.model.model import Decision


def determine_winner(initial_scores: List[Decision], final_scores: List[Decision]) -> str:
    initial_map = {score["name"]: score["value"] for score in initial_scores}

    swing_agree = 0
    swing_disagree = 0

    for decision in final_scores:
        name = decision["name"]
        final_vote = decision["value"]
        initial_vote = initial_map.get(name)

        if initial_vote is not None and initial_vote != final_vote:
            if final_vote == "agree":
                swing_agree += 1
            elif final_vote == "disagree":
                swing_disagree += 1

    if swing_agree > swing_disagree:
        return "Pro Team wins"
    elif swing_disagree > swing_agree:
        return "Opposing Team wins"
    else:
        return "Tie"


def audience_init_node(state: DebateState):
    audience_workflow = create_audience("init", state["audience_members"])
    audience = audience_workflow.compile()
    result = audience.invoke({
        "topic": state["topic"],
        "transcript": [],
        "initial_scores": [],
        "final_scores": []
    })

    state["initial_scores"] = result["initial_scores"]

    return state


def audience_final_node(state: DebateState):
    audience_workflow = create_audience("final", state["audience_members"])
    audience = audience_workflow.compile()
    result = audience.invoke({
        "topic": state["topic"],
        "transcript": state["transcript"],
        "initial_scores": [],
        "final_scores": []
    })

    state["final_scores"] = result["final_scores"]

    print(determine_winner(state["initial_scores"], state["final_scores"]))

    return state


def pro_team_node(state: DebateState):
    team_workflow = create_team_workflow(state["proposing_members"])
    team = team_workflow.compile()
    result = team.invoke({
        "topic": state["topic"],
        "team_role": TeamRole.PROPOSING,
        "transcript": state["transcript"],
        "audience_profile": {
             "audience_members": state["audience_members"]
        }
    })
    state["round"] += 1
    state["transcript"] = result["transcript"]
    return state


def opp_team_node(state: DebateState):
    team_workflow = create_team_workflow(state["opposing_members"])
    team = team_workflow.compile()
    result = team.invoke({
        "topic": state["topic"],
        "team_role": TeamRole.OPPOSING,
        "transcript": state["transcript"],
        "audience_profile": {
            "audience_members": state["audience_members"]
        }
    })
    state["round"] += 1
    state["transcript"] = result["transcript"]
    return state


choice = random.choice([True, False])
first_team = "pro_team" if choice else "opp_team"
second_team = "opp_team" if choice else "pro_team"


def next_round(state: DebateState):
    if "round" in state and state["round"] / 2 < 1:
        return first_team
    else:
        return "audience_final"


def create_debate():
    workflow = StateGraph(DebateState)
    workflow.add_node("audience_init", audience_init_node)
    workflow.add_node("pro_team", pro_team_node)
    workflow.add_node("opp_team", opp_team_node)
    workflow.add_node("audience_final", audience_final_node)

    workflow.add_edge(START, "audience_init")
    workflow.add_edge("audience_init", first_team)
    workflow.add_edge(first_team, second_team)
    workflow.add_conditional_edges(second_team, next_round, [first_team, "audience_final"])
    workflow.add_edge("audience_final", END)

    return workflow
