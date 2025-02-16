from typing_extensions import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.tutor.state import TutorState
from agents.model.model import Transcript
from agents.hub import gpt_4o_mini
from agents.tutor.utils import summarize_audience_profile


class FactualAnalysisOutput(BaseModel):
    factual_evaluation: str = Field(
        description="A summary of the overall factual accuracy assessment."
    )
    evidence_quality: str = Field(
        description="Evaluation of the credibility and reliability of the evidence provided."
    )
    missing_sources: str = Field(
        description="Identification of any missing citations or unsupported claims."
    )


def modify_transcripts(transcript: List[Transcript]) -> str:
    return "\n".join(
        [f"{t['speaker']} ({t['team_role']}): {t['text']}" for t in transcript]
    )


def factual_analysis(state: TutorState):
    user_args = modify_transcripts(state["user_arguments"])
    opponent_args = modify_transcripts(state["opponent_arguments"])
    combined_content = (
        f"User Arguments:\n{user_args}\n\nOpponent Arguments:\n{opponent_args}"
    )

    factual_analysis_prompt = PromptTemplate(
        template="""
            Role:
            You are a **Factual Analysis Agent**. Your task is to evaluate the factual accuracy of the debate on the topic "{topic}".
            Below are the arguments presented by both sides:
            
            {combined_content}
            
            Audience Profile:
            {audience_profile}
            
            Please perform the following:
            1. Assess the overall factual accuracy of the arguments.
            2. Evaluate the quality and credibility of the evidence provided.
            3. Identify any missing citations or unsupported claims.
            
            Return your analysis in JSON format with these keys:
            - "factual_evaluation": A summary of factual accuracy.
            - "evidence_quality": An evaluation of the supporting evidence.
            - "missing_sources": Details on any missing or unverified sources.
        """,
        input_variables=["topic", "transcript", "audience_profile"]
    )

    factual_analysis_chain = factual_analysis_prompt | gpt_4o_mini.with_structured_output(FactualAnalysisOutput)
    analysis_result = factual_analysis_chain.invoke({
        "topic": state["topic"],
        "combined_content": combined_content,
        "audience_profile": summarize_audience_profile(state["audience_profile"]),
    })

    return analysis_result


class FactualFeedbackOutput(BaseModel):
    score: int = Field(
        description="Numeric factual accuracy score (1-10) reflecting the strength of the evidence."
    )
    feedback: str = Field(
        description="Actionable suggestions to improve the factual support of the arguments."
    )


def factual_accuracy_node(state: TutorState) -> TutorState:
    analysis: FactualAnalysisOutput = factual_analysis(state)

    factual_feedback_prompt = PromptTemplate(
        template="""
            Role:
            You are a **Factual Feedback Agent**. Based on the following factual analysis:
            
            - Factual Evaluation: {factual_evaluation}
            - Evidence Quality: {evidence_quality}
            - Missing Sources: {missing_sources}
            
            Please provide:
            1. A numeric factual accuracy score (from 1 to 10).
            2. Actionable feedback on how to improve the factual support of the arguments.
            
            Return your response in JSON format with these keys:
            - "score": The numeric factual accuracy score.
            - "feedback": Detailed suggestions for improvement.
            """,
        input_variables=["factual_evaluation", "evidence_quality", "missing_sources"]
    )

    factual_feedback_chain = factual_feedback_prompt | gpt_4o_mini.with_structured_output(FactualFeedbackOutput)
    feedback_result = factual_feedback_chain.invoke({
        "factual_evaluation": analysis.factual_evaluation,
        "evidence_quality": analysis.evidence_quality,
        "missing_sources": analysis.missing_sources,
    })

    state["factual_analysis"] = {
        "score": feedback_result.score,
        "feedback": feedback_result.feedback,
    }

    return state
