from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(find_dotenv())


llm = ChatOpenAI(model="gpt-4o-mini")


class DebateState(TypedDict):
    statement: str
    proponent_arguments: str
    opponent_arguments: str
    result: str


class Proponent(BaseModel):
    proponent_arguments: str = Field(description="The proponent arguments for statement.")


class Opponent(BaseModel):
    opponent_arguments: str = Field(description="The opponent arguments against statement")


class JudgeScore(BaseModel):
    proponent_score: str = Field(description="Score for the proponent.")
    opponent_score: str = Field(description="Score for the opponent.")
    reasoning: str = Field(description="Reasoning behind the scores.")


proponent_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a debating agent arguing whether a statement is true."
        ),
        (
            "human", "Statement: {statement}"
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
            "ai", "Proponent arguments: {proponent_arguments}"
        )
    ]
)

judge_template = ChatPromptTemplate.from_messages(
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

proponent_chain = proponent_template | llm.with_structured_output(Proponent)
opponent_chain = opponent_template | llm.with_structured_output(Opponent)
judge_chain = judge_template | llm.with_structured_output(JudgeScore)


def proponent_node(state: DebateState):
    proponent_arguments = proponent_chain.invoke(state)
    return {
        "proponent_arguments": proponent_arguments,
    }


def opponent_node(state: DebateState):
    opponent_arguments = opponent_chain.invoke(state)
    return {
        "opponent_arguments": opponent_arguments,
    }


def judge_node(state: DebateState):
    output = judge_chain.invoke(state)
    return {
        "result": f"Proponent: {output.proponent_score} Opponent: {output.opponent_score} Reasoning: {output.reasoning}",
    }


builder = StateGraph(DebateState)

builder.add_node("proponent", proponent_node)
builder.add_node("opponent", opponent_node)
builder.add_node("judge", judge_node)

builder.add_edge(START, "proponent")
builder.add_edge("proponent", "opponent")
builder.add_edge("opponent", "judge")

builder.add_edge("judge", END)

debate_graph = builder.compile()

debate_graph.get_graph().draw_mermaid_png(output_file_path ="output_graph.png")


def run_debate(statement):
    state = {
        "statement": statement
    }

    events = debate_graph.stream(state, stream_mode="values")

    for event in events:
        print(event)


if __name__ == "__main__":
    run_debate("The best football player is Messi")
