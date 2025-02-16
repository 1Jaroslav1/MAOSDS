from typing_extensions import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from agents.tutor.state import TutorState
from agents.tutor.utils import summarize_audience_profile


class FallaciesAnalysisOutput(BaseModel):
    fallacies_detected: str = Field(
        description="A summary of the logical fallacies detected in the arguments (e.g., slippery slope, false dilemma, hasty generalization, circular reasoning, etc.)."
    )
    pitfall_observations: str = Field(
        description="Additional observations regarding the frequency or severity of the detected fallacies."
    )


class FallaciesFeedbackOutput(BaseModel):
    score: int = Field(
        description="Numeric score (1-10) reflecting the overall severity and frequency of logical fallacies."
    )
    feedback: str = Field(
        description="Actionable feedback with suggestions on how to address or avoid these logical fallacies."
    )


def modify_transcripts(transcripts: List[dict]) -> str:
    return "\n".join(
        [f"{t['speaker']} ({t['team_role']}): {t['text']}" for t in transcripts]
    )


def logical_fallacies_analysis(state: TutorState) -> FallaciesAnalysisOutput:
    user_args = modify_transcripts(state["user_arguments"])
    opponent_args = modify_transcripts(state["opponent_arguments"])
    combined_content = (
        f"User Arguments:\n{user_args}\n\nOpponent Arguments:\n{opponent_args}"
    )

    fallacies_analysis_prompt = PromptTemplate(
        template="""
            Role:
            You are a **Logical Fallacies & Pitfalls Detector**. Your task is to analyze the debate on the topic "{topic}" and identify any logical fallacies present in the arguments. These may include, but are not limited to, slippery slope, false dilemma, hasty generalization, circular reasoning, and weak analogies.
            
            Below are the combined arguments from both sides:
            
            {combined_content}
            
            Audience Profile:
            {audience_profile}
            
            Please perform the following tasks:
            1. Identify any logical fallacies present in the arguments.
            2. Provide a brief summary or definition for each detected fallacy.
            3. Include any observations about the severity or frequency of these fallacies.
            
            Return your analysis in JSON format with the following keys:
            - "fallacies_detected": A summary of the logical fallacies detected.
            - "pitfall_observations": Additional observations regarding the logical pitfalls.
            """,
        input_variables=["topic", "combined_content", "audience_profile"]
    )

    fallacies_analysis_chain = fallacies_analysis_prompt | gpt_4o_mini.with_structured_output(FallaciesAnalysisOutput)
    analysis_result = fallacies_analysis_chain.invoke({
        "topic": state["topic"],
        "combined_content": combined_content,
        "audience_profile": summarize_audience_profile(state["audience_profile"]),
    })

    return analysis_result


def logical_fallacies_feedback_node(state: TutorState) -> TutorState:
    analysis = logical_fallacies_analysis(state)

    fallacies_feedback_prompt = PromptTemplate(
        template="""
        Role:
        You are a **Logical Fallacies Feedback Agent**. Based on the following analysis of the debate:
        
        - Fallacies Detected: {fallacies_detected}
        - Pitfall Observations: {pitfall_observations}
        
        Please provide:
        1. A numeric score (1–10) reflecting the overall presence and severity of logical fallacies:
           - 1–3: Multiple glaring fallacies that undermine the argument.
           - 4–6: Some fallacies present, but the main argument remains somewhat intact.
           - 7–9: Generally free of major fallacies; might have minor errors or weak analogies.
           - 10: No recognizable fallacies; strong, logically sound arguments.
        2. Actionable feedback with suggestions on how to address or avoid these fallacies in future arguments.
        
        Return your response in JSON format with these keys:
        - "score": The numeric score.
        - "feedback": Detailed suggestions for improvement.
        """,
        input_variables=["fallacies_detected", "pitfall_observations"]
    )

    fallacies_feedback_chain = fallacies_feedback_prompt | gpt_4o_mini.with_structured_output(FallaciesFeedbackOutput)
    feedback_result = fallacies_feedback_chain.invoke({
        "fallacies_detected": analysis.fallacies_detected,
        "pitfall_observations": analysis.pitfall_observations,
    })

    state["logical_fallacies"] = {
        "score": feedback_result.score,
        "feedback": feedback_result.feedback,
    }

    return state
