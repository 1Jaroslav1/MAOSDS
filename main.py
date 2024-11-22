from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(find_dotenv())


llm = ChatOpenAI(model="gpt-4o-mini")


class Round(TypedDict):
    round: int
    proponent_arguments: str
    opponent_arguments: str
    proponent_score: int
    opponent_score: int
    reasoning: str


class DebateState(TypedDict):
    statement: str
    rounds: List[Round]
    result: str


class Proponent(BaseModel):
    proponent_arguments: str = Field(description="The proponent arguments for statement.")


class Opponent(BaseModel):
    opponent_arguments: str = Field(description="The opponent arguments against statement")


class RoundJudgeScore(BaseModel):
    proponent_score: int = Field(ge=0, le=10, description="Score for the proponent.")
    opponent_score: int = Field(ge=0, le=10, description="Score for the opponent.")
    reasoning: str = Field(description="Reasoning behind the scores.")


class SuperJudgeState(BaseModel):
    proponent_score: int = Field(ge=0, le=10, description="Score for the proponent.")
    opponent_score: int = Field(ge=0, le=10, description="Score for the opponent.")
    reasoning: str = Field(description="Reasoning behind the scores.")


proponent_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a debating agent arguing whether a statement is true."
        ),
        (
            "human", "Statement: {statement}"
        ),
        (
            "ai", "You have currently said such arguments: {prev_proponent_arguments}"
        ),
        (
            "ai", "You opponent have currently said such counterarguments: {prev_opponent_arguments}"
        )
    ]
)

opponent_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a debating agent arguing whether a statement is false."
        ),
        (
            "human", "Statement: {statement}"
        ),
        (
            "ai", "You have currently said such arguments: {prev_opponent_arguments}"
        ),
        (
            "ai", "You opponent have currently said such counterarguments: {prev_proponent_arguments}"
        )
    ]
)

round_judge_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a judge evaluating arguments in a debate. Score the arguments based on their quality, coherence, and relevance."
        ),
        (
            "human", "Statement: {statement}"
        ),
        (
            "ai", "Proponent arguments: {proponent_arguments}"
        ),
        (
            "ai", "Opponent arguments: {opponent_arguments}"
        )
    ]
)

super_judge_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a super judge evaluating the results of the debate according to your colleagues (round judges) results"
        ),
        (
            "human", "Statement: {statement}"
        ),
        (
            "ai", "Rounds results: {rounds}"
        ),
    ]
)

proponent_chain = proponent_template | llm.with_structured_output(Proponent)
opponent_chain = opponent_template | llm.with_structured_output(Opponent)
round_judge_chain = round_judge_template | llm.with_structured_output(RoundJudgeScore)
super_judge_chain = super_judge_template | llm.with_structured_output(SuperJudgeState)


def proponent_node(state: DebateState):
    output = proponent_chain.invoke(
        {
            "statement": state["statement"],
            "prev_proponent_arguments": [round_["proponent_arguments"] for round_ in state["rounds"]],
            "prev_opponent_arguments": [round_["opponent_arguments"] for round_ in state["rounds"]]
        }
    )
    rounds = state.get("rounds")
    new_round: Round = {
        "round": len(rounds) + 1,
        "proponent_arguments": output.proponent_arguments,
        "opponent_arguments": "",
        "proponent_score": 0,
        "opponent_score": 0,
        "reasoning": ""
    }
    rounds.append(new_round)
    return {
        "rounds": rounds
    }


def opponent_node(state: DebateState):
    output = opponent_chain.invoke(
        {
            "statement": state["statement"],
            "prev_proponent_arguments": [round_["proponent_arguments"] for round_ in state["rounds"]],
            "prev_opponent_arguments": [round_["opponent_arguments"] for round_ in state["rounds"]]
        }
    )
    rounds = state.get("rounds")
    new_round: Round = rounds[-1]
    new_round["opponent_arguments"] = output.opponent_arguments
    rounds[-1] = new_round
    return {
        "rounds": rounds
    }


def round_judge_node(state: DebateState):
    output = round_judge_chain.invoke(
        {
            "statement": state["statement"],
            "proponent_arguments": state["rounds"][-1]["proponent_arguments"],
            "opponent_arguments": state["rounds"][-1]["opponent_arguments"],
        }
    )
    rounds = state.get("rounds")
    new_round: Round = rounds[-1]
    new_round["proponent_score"] = output.proponent_score
    new_round["opponent_score"] = output.opponent_score
    new_round["reasoning"] = output.reasoning
    rounds[-1] = new_round
    return {
        "rounds": rounds
    }


def super_judge_node(state: DebateState):
    output = super_judge_chain.invoke(state)
    return {
        "result": f"Proponent: {output.proponent_score} Opponent: {output.opponent_score} Reasoning: {output.reasoning}",
    }


def should_move_to_super_judge(state: DebateState):
    if "rounds" in state and len(state["rounds"]) > 2:
        return "super_judge"
    else:
        return "proponent"


builder = StateGraph(DebateState)

builder.add_node("proponent", proponent_node)
builder.add_node("opponent", opponent_node)
builder.add_node("round_judge", round_judge_node)
builder.add_node("super_judge", super_judge_node)

builder.add_edge(START, "proponent")
builder.add_edge("proponent", "opponent")
builder.add_edge("opponent", "round_judge")
builder.add_conditional_edges("round_judge", should_move_to_super_judge, ["proponent", "super_judge"])
builder.add_edge("super_judge", END)

debate_graph = builder.compile()

debate_graph.get_graph().draw_mermaid_png(output_file_path ="output_graph.png")


def run_debate(statement):
    state: DebateState = {
        "statement": statement,
        "rounds": [],
        "result": ""
    }

    events = debate_graph.stream(state, stream_mode="values")

    for event in events:
        print(event)


if __name__ == "__main__":
    # run_debate("The best football player is Messi")
    run_debate("The best football player is Kylian Mbappé")
