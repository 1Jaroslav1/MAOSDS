from typing_extensions import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from agents.tutor.state import TutorState
from agents.tutor.utils import summarize_audience_profile


class LogicalAnalysisOutput(BaseModel):
    structure_evaluation: str = Field(
        description="A summary of the evaluation of the logical structure, including the presence of a thesis, premises, and conclusion."
    )
    flow_assessment: str = Field(
        description="An evaluation of the clarity and coherence of the argument flow, noting any gaps or unclear transitions."
    )
    transition_observations: str = Field(
        description="Observations on the use of transitional phrases and handling of counterarguments."
    )


class LogicalFeedbackOutput(BaseModel):
    score: int = Field(
        description="Numeric score (1-10) reflecting the overall logical structure and coherence."
    )
    feedback: str = Field(
        description="Actionable suggestions to improve the logical flow and organization of the arguments."
    )


def modify_transcripts(transcripts: List[dict]) -> str:
    return "\n".join(
        [f"{t['speaker']} ({t['team_role']}): {t['text']}" for t in transcripts]
    )


def logical_structure_analysis(state: TutorState) -> LogicalAnalysisOutput:
    user_args = modify_transcripts(state["user_arguments"])
    opponent_args = modify_transcripts(state["opponent_arguments"])
    combined_content = (
        f"User Arguments:\n{user_args}\n\nOpponent Arguments:\n{opponent_args}"
    )

    logical_analysis_prompt = PromptTemplate(
        template="""
            Role:
            You are a **Logical Structure & Coherence Agent**. Your task is to evaluate the logical organization of the debate on the topic "{topic}".
            Below are the arguments presented by both sides:
            
            {combined_content}
            
            Audience Profile:
            {audience_profile}
            
            Please perform the following tasks:
            1. Assess the overall logical structure of the arguments, including whether there is a clear thesis, supporting premises, and conclusion.
            2. Evaluate the clarity and coherence of the argument flow, noting any gaps or unclear transitions.
            3. Comment on the use of transitional phrases and how potential counterarguments are addressed.
            
            Return your analysis in JSON format with these keys:
            - "structure_evaluation": A summary of the logical structure assessment.
            - "flow_assessment": An evaluation of the clarity and coherence of the argument flow.
            - "transition_observations": Observations on the use of transitional phrases or handling of counterarguments.
            """,
        input_variables=["topic", "combined_content", "audience_profile"]
    )

    logical_analysis_chain = logical_analysis_prompt | gpt_4o_mini.with_structured_output(LogicalAnalysisOutput)
    analysis_result = logical_analysis_chain.invoke({
        "topic": state["topic"],
        "combined_content": combined_content,
        "audience_profile": summarize_audience_profile(state["audience_profile"]),
    })

    return analysis_result


def logical_structure_feedback_node(state: TutorState) -> TutorState:
    analysis = logical_structure_analysis(state)

    logical_feedback_prompt = PromptTemplate(
        template="""
            Role:
            You are a **Logical Structure Feedback Agent**. Based on the following analysis:
            
            - Structure Evaluation: {structure_evaluation}
            - Flow Assessment: {flow_assessment}
            - Transition Observations: {transition_observations}
            
            Please provide:
            1. A numeric score (1–10) reflecting the overall logical structure and coherence of the arguments:
               - 1–3: Disorganized argument, little to no logical flow.
               - 4–6: Some logical structure but with gaps or unclear transitions.
               - 7–9: Generally coherent and well-organized with only minor issues.
               - 10: Excellent flow; arguments are clearly layered and easy to follow.
            2. Actionable feedback with suggestions for improving the logical organization, such as using transitional phrases or grouping related ideas.
            
            Return your response in JSON format with these keys:
            - "score": The numeric score.
            - "feedback": Detailed suggestions for improvement.
            """,
        input_variables=["structure_evaluation", "flow_assessment", "transition_observations"]
    )

    logical_feedback_chain = logical_feedback_prompt | gpt_4o_mini.with_structured_output(LogicalFeedbackOutput)
    feedback_result = logical_feedback_chain.invoke({
        "structure_evaluation": analysis.structure_evaluation,
        "flow_assessment": analysis.flow_assessment,
        "transition_observations": analysis.transition_observations,
    })

    state["logical_structure"] = {
        "score": feedback_result.score,
        "feedback": feedback_result.feedback,
    }

    return state

