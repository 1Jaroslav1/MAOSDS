from typing_extensions import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from agents.tutor.state import TutorState
from agents.tutor.utils import summarize_audience_profile


class StyleClarityAnalysisOutput(BaseModel):
    clarity_assessment: str = Field(
        description="Evaluation of the clarity and conciseness of the language used in the arguments."
    )
    rhetorical_device_evaluation: str = Field(
        description="Assessment of the use of rhetorical devices (e.g., rhetorical questions, imagery, metaphors) and overall stylistic flair."
    )
    readability_assessment: str = Field(
        description="Evaluation of overall readability, pacing, and language precision in relation to the audience."
    )


class StyleClarityFeedbackOutput(BaseModel):
    score: int = Field(
        description="Numeric score (1-10) reflecting the overall style, clarity, and rhetorical effectiveness."
    )
    feedback: str = Field(
        description="Actionable suggestions for enhancing clarity, conciseness, and overall rhetorical style."
    )


def modify_transcripts(transcripts: List[dict]) -> str:
    return "\n".join(
        [f"{t['speaker']} ({t['team_role']}): {t['text']}" for t in transcripts]
    )


def style_clarity_analysis(state: TutorState) -> StyleClarityAnalysisOutput:
    user_args = modify_transcripts(state["user_arguments"])
    opponent_args = modify_transcripts(state["opponent_arguments"])
    combined_content = f"User Arguments:\n{user_args}\n\nOpponent Arguments:\n{opponent_args}"

    style_clarity_prompt = PromptTemplate(
        template="""
            Role:
            You are a **Style & Clarity (Rhetorical Technique) Agent**. Your task is to evaluate the overall clarity, conciseness, and rhetorical style of the debate on the topic "{topic}".
            Below are the combined arguments from both sides:
            
            {combined_content}
            
            Audience Profile:
            {audience_profile}
            
            Please perform the following tasks:
            1. Evaluate the clarity and conciseness of the language used. Are the arguments easy to follow, or are they overly verbose or vague?
            2. Assess the use of rhetorical devices such as rhetorical questions, imagery, and metaphors. Comment on the overall stylistic flair.
            3. Evaluate the overall readability, pacing, and precision of language, considering the audience’s familiarity with the topic.
            
            Return your analysis in JSON format with these keys:
            - "clarity_assessment": Your evaluation of clarity and conciseness.
            - "rhetorical_device_evaluation": Your assessment of the use of rhetorical devices and stylistic flair.
            - "readability_assessment": Your evaluation of overall readability and language precision.
            """,
        input_variables=["topic", "combined_content", "audience_profile"]
    )

    style_clarity_chain = style_clarity_prompt | gpt_4o_mini.with_structured_output(StyleClarityAnalysisOutput)
    analysis_result = style_clarity_chain.invoke({
        "topic": state["topic"],
        "combined_content": combined_content,
        "audience_profile": summarize_audience_profile(state["audience_profile"]),
    })

    return analysis_result


def style_clarity_feedback_node(state: TutorState) -> TutorState:
    analysis = style_clarity_analysis(state)

    style_clarity_feedback_prompt = PromptTemplate(
        template="""
        Role:
        You are a **Style & Clarity Feedback Agent**. Based on the following analysis:
        
        - Clarity Assessment: {clarity_assessment}
        - Rhetorical Device Evaluation: {rhetorical_device_evaluation}
        - Readability Assessment: {readability_assessment}
        
        Please provide:
        1. A numeric score (1–10) reflecting the overall style, clarity, and rhetorical effectiveness:
           - 1–3: Hard to follow, overly verbose or vague.
           - 4–6: Understandable but stylistically inconsistent; frequent filler phrases.
           - 7–9: Generally clear, engaging language with minimal stylistic issues.
           - 10: Captivating style and precise language, well-suited to the topic and audience.
        2. Actionable feedback with recommendations for simplifying or enhancing word choice and overall presentation, referencing the audience profile where applicable.
        
        Return your response in JSON format with these keys:
        - "score": The numeric score.
        - "feedback": Detailed suggestions for improvement.
        """,
        input_variables=["clarity_assessment", "rhetorical_device_evaluation", "readability_assessment"]
    )

    style_clarity_feedback_chain = style_clarity_feedback_prompt | gpt_4o_mini.with_structured_output(
        StyleClarityFeedbackOutput)
    feedback_result = style_clarity_feedback_chain.invoke({
        "clarity_assessment": analysis.clarity_assessment,
        "rhetorical_device_evaluation": analysis.rhetorical_device_evaluation,
        "readability_assessment": analysis.readability_assessment,
    })

    state["style_clarity"] = {
        "score": feedback_result.score,
        "feedback": feedback_result.feedback,
    }

    return state
