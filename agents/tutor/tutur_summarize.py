from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from agents.tutor.state import TutorState


class ComplexFeedbackOutput(BaseModel):
    overall_score: int = Field(
        description="Aggregated overall score (1-10) considering all evaluation dimensions."
    )
    detailed_feedback: str = Field(
        description="A comprehensive summary of the evaluations, highlighting strengths, weaknesses, and actionable recommendations."
    )


def summary_feedback_node(state: TutorState) -> TutorState:
    summary_feedback_prompt = PromptTemplate(
        template="""
            You are a **Comprehensive Feedback Summarizer Agent**. You have received the following feedback from five different evaluation agents:
            
            Factual Analysis:
              - Score: {factual_score}
              - Feedback: {factual_feedback}
            
            Logical Structure:
              - Score: {logical_structure_score}
              - Feedback: {logical_structure_feedback}
            
            Logical Fallacies:
              - Score: {logical_fallacies_score}
              - Feedback: {logical_fallacies_feedback}
            
            Emotional Appeal:
              - Score: {emotional_appeal_score}
              - Feedback: {emotional_appeal_feedback}
            
            Style & Clarity:
              - Score: {style_clarity_score}
              - Feedback: {style_clarity_feedback}
            
            Based on the above evaluations, please provide:
            1. An overall aggregated score (1-10) that considers the strengths and weaknesses across all dimensions.
            2. A comprehensive summary that synthesizes the individual feedback into detailed, actionable recommendations for improvement.
            
            Return your response in JSON format with the following keys:
            - "overall_score": The aggregated overall score.
            - "detailed_feedback": A comprehensive summary of the feedback.
                    """,
        input_variables=[
            "factual_score", "factual_feedback",
            "logical_structure_score", "logical_structure_feedback",
            "logical_fallacies_score", "logical_fallacies_feedback",
            "emotional_appeal_score", "emotional_appeal_feedback",
            "style_clarity_score", "style_clarity_feedback",
        ]
    )

    summary_feedback_chain = summary_feedback_prompt | gpt_4o_mini.with_structured_output(ComplexFeedbackOutput)
    summary_result = summary_feedback_chain.invoke({
        "factual_score": state["factual_analysis"]["score"],
        "factual_feedback": state["factual_analysis"]["feedback"],
        "logical_structure_score": state["logical_structure"]["score"],
        "logical_structure_feedback": state["logical_structure"]["feedback"],
        "logical_fallacies_score": state["logical_fallacies"]["score"],
        "logical_fallacies_feedback": state["logical_fallacies"]["feedback"],
        "emotional_appeal_score": state["emotional_appeal"]["score"],
        "emotional_appeal_feedback": state["emotional_appeal"]["feedback"],
        "style_clarity_score": state["style_clarity"]["score"],
        "style_clarity_feedback": state["style_clarity"]["feedback"],
    })

    state["complex_feedback"] = {
        "score": summary_result.overall_score,
        "feedback": summary_result.detailed_feedback,
    }

    return state
