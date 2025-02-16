from typing_extensions import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from agents.tutor.state import TutorState
from agents.tutor.utils import summarize_audience_profile


class EmotionalAnalysisOutput(BaseModel):
    emotional_tone: str = Field(
        description="Evaluation of the overall emotional tone of the arguments, including whether the emotional appeals are evocative, manipulative, or balanced."
    )
    narrative_strength: str = Field(
        description="Assessment of the storytelling elements, such as the use of personal anecdotes, vivid examples, and narrative flow."
    )
    balance_assessment: str = Field(
        description="Evaluation of how well emotional appeals are integrated with logical, data-driven content."
    )


class EmotionalFeedbackOutput(BaseModel):
    score: int = Field(
        description="Numeric score (1–10) reflecting the overall effectiveness of the emotional appeal and storytelling."
    )
    feedback: str = Field(
        description="Actionable suggestions to improve emotional engagement and storytelling, considering the audience profile if provided."
    )


def modify_transcripts(transcripts: List[dict]) -> str:
    return "\n".join(
        [f"{t['speaker']} ({t['team_role']}): {t['text']}" for t in transcripts]
    )


def emotional_analysis(state: TutorState) -> EmotionalAnalysisOutput:
    user_args = modify_transcripts(state["user_arguments"])
    opponent_args = modify_transcripts(state["opponent_arguments"])
    combined_content = f"User Arguments:\n{user_args}\n\nOpponent Arguments:\n{opponent_args}"

    emotional_analysis_prompt = PromptTemplate(
        template="""
        Role:
        You are an **Emotional Appeal & Storytelling Agent**. Your task is to analyze the use of emotional appeals, narratives, and personal anecdotes in the debate on the topic "{topic}".
        Below are the combined arguments from both sides:
        
        {combined_content}
        
        Audience Profile:
        {audience_profile}
        
        Please perform the following tasks:
        1. Evaluate the overall emotional tone of the arguments. Are they appropriately evocative or overly manipulative?
        2. Assess the strength and clarity of the storytelling elements, including the use of personal anecdotes and vivid examples.
        3. Determine how well the emotional appeals are integrated with logical, data-driven content.
        4. Consider the relevance of the emotional content for the given audience profile.
        
        Return your analysis in JSON format with these keys:
        - "emotional_tone": Your evaluation of the overall emotional tone.
        - "narrative_strength": Your assessment of the storytelling and narrative elements.
        - "balance_assessment": Your evaluation of the balance between emotional and rational content.
        """,
        input_variables=["topic", "combined_content", "audience_profile"]
    )

    emotional_analysis_chain = emotional_analysis_prompt | gpt_4o_mini.with_structured_output(EmotionalAnalysisOutput)
    analysis_result = emotional_analysis_chain.invoke({
        "topic": state["topic"],
        "combined_content": combined_content,
        "audience_profile": summarize_audience_profile(state["audience_profile"]),
    })

    return analysis_result


def emotional_feedback_node(state: TutorState) -> TutorState:
    analysis = emotional_analysis(state)

    emotional_feedback_prompt = PromptTemplate(
        template="""
            Role:
            You are an **Emotional Appeal Feedback Agent**. Based on the following analysis:
            
            - Emotional Tone: {emotional_tone}
            - Narrative Strength: {narrative_strength}
            - Balance Assessment: {balance_assessment}
            
            Please provide:
            1. A numeric score (1–10) reflecting the overall effectiveness of the emotional appeal and storytelling:
               - 1–3: Overly emotional or off-topic; emotional appeals distract rather than support.
               - 4–6: Some emotional stories, but lack relevance or integration with main points.
               - 7–9: Effective use of anecdotes, balanced with logical argumentation.
               - 10: Masterful storytelling that reinforces the argument powerfully without overshadowing facts.
            2. Actionable feedback with suggestions on how to enhance the emotional engagement and storytelling, especially in light of the audience profile.
            
            Return your response in JSON format with these keys:
            - "score": The numeric score.
            - "feedback": Detailed suggestions for improvement.
            """,
        input_variables=["emotional_tone", "narrative_strength", "balance_assessment"]
    )

    emotional_feedback_chain = emotional_feedback_prompt | gpt_4o_mini.with_structured_output(EmotionalFeedbackOutput)
    feedback_result = emotional_feedback_chain.invoke({
        "emotional_tone": analysis.emotional_tone,
        "narrative_strength": analysis.narrative_strength,
        "balance_assessment": analysis.balance_assessment,
    })

    state["emotional_appeal"] = {
        "score": feedback_result.score,
        "feedback": feedback_result.feedback,
    }

    return state
