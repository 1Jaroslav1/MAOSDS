from langchain_core.prompts import PromptTemplate
from state import TeamMemberState, TeamRole
from agents.hub import gpt_4o_mini
from pydantic import BaseModel, Field
from typing_extensions import List


class OpponentArgumentsOutput(BaseModel):
    extracted_arguments: List[str] = Field(
        description="List of extracted key arguments presented by the opposing team."
    )


class AnalysisNodeOutput(BaseModel):
    main_themes_and_issues: List[str] = Field("Main themes and issues")
    opponent_perspectives: List[str] = Field("Opponents’ likely perspectives")
    opponent_weaknesses: List[str] = Field("Weaknesses or inconsistencies in the opponent’s arguments")


def opponent_team_role(team_role: TeamRole) -> TeamRole:
    return TeamRole.OPPOSING if team_role == TeamRole.PROPOSING else TeamRole.PROPOSING


def extract_team_arguments(state: TeamMemberState) -> List[str]:
    return [
        message["text"]
        for message in state["transcript"]
        if message["team_role"] == state["team_role"]
    ]


def extract_opponent_arguments(state: TeamMemberState, opponent_team_role: TeamRole) -> List[str]:
    opponent_transcript = [
        message["text"] for message in state["transcript"] if message["team_role"] == opponent_team_role
    ]

    if not opponent_transcript:
        return []

    prompt = PromptTemplate(
        template="""
            Role:
            You are the Opponent Arguments Extractor Agent.
            You should extract opponent arguments like you are human with such personality and experience: {person}
            You are in {team_role} team.
            
            Your task is to extract and structure key arguments from the opponent team’s statements.

            Instructions:
            - Identify **clear and concise** argument points from the transcript.
            - Ignore rhetorical statements, general commentary, and moderator inputs.
            - Format arguments in bullet points, capturing **only substantive claims**.
            
            Topic: {topic}
            Debate Transcript:
            {opponent_transcript}

            Extracted Arguments:
            - (Provide a structured list of arguments)
        """,
        input_variables=["topic", "team_role", "person", "opponent_transcript"]
    )

    chain = prompt | gpt_4o_mini.with_structured_output(OpponentArgumentsOutput)
    result = chain.invoke({
        "topic": state["topic"],
        "team_role": state["team_role"],
        "person": state["person"],
        "opponent_transcript": opponent_transcript
    })

    return result.extracted_arguments


def analysis_node(state: TeamMemberState) -> TeamMemberState:
    state["opponent_arguments"] = extract_opponent_arguments(state, opponent_team_role(state["team_role"]))
    state["team_arguments"] = extract_team_arguments(state)

    prompt = PromptTemplate(
        template="""
            Role:
            You are the Analyzer Agent.
            You should analyze like you are human with such personality and experience: {person}
            Your task is to examine the following debate context—including the transcript, opponent’s arguments, and audience profile—and perform the following analyses:
              - Summarize the main themes and issues.
              - Evaluate the arguments already provided by your teammates and identify gaps or opportunities for improvement.
              - Infer the opponents’ likely perspectives.
              - Highlight any weaknesses or inconsistencies in the opponent’s arguments.
            
            Additional Instructions:
            - This is a reprocessing cycle based on evaluator feedback. Please review the previous evaluation summary and suggestions:
              Evaluation Summary: {evaluation_summary}
              Suggestions: {evaluation_suggestions}
            - Re-assess the transcript and opponent arguments to capture any new insights or address previously noted shortcomings.
            
            Debate Context:
            1. Topic: {topic}
            2. Transcript: {transcript}
            3. Opponent Arguments: {opponent_arguments}
            4. Team's Previous Arguments: {team_arguments}
            5. Audience Profile: {audience_profile}

            Tasks:
            1. Provide your output in a structured format with bullet points summarizing your key insights.
        """,
        input_variables=["topic", "person", "transcript", "opponent_arguments", "team_arguments", "audience_profile", "evaluation_summary", "evaluation_suggestions"]
    )
    chain = prompt | gpt_4o_mini.with_structured_output(AnalysisNodeOutput)
    result = chain.invoke({
        "topic": state["topic"],
        "person": state["person"],
        "transcript": state["transcript"],
        "opponent_arguments": state["opponent_arguments"],
        "team_arguments": state["team_arguments"],
        "audience_profile": state["audience_profile"],
        "evaluation_summary": state["evaluation"].get("evaluation_summary", "No previous evaluation available."),
        "evaluation_suggestions": state["evaluation"].get("suggestions", "No suggestions provided.")
    })

    state["analysis"] = {
        "main_themes_and_issues": result.main_themes_and_issues,
        "opponent_perspectives": result.opponent_perspectives,
        "opponent_weaknesses": result.opponent_weaknesses
    }

    return state
