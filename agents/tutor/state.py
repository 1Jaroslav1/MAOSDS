from typing_extensions import TypedDict, List
from agents.model.model import TeamRole, Transcript, AudienceProfile
from typing import Annotated

def reducer(existing, new):
    return existing if existing else new


class Feedback(TypedDict):
    score: int
    feedback: str


class TutorState(TypedDict):
    topic: Annotated[str, reducer]
    user_arguments: Annotated[List[Transcript], reducer]
    opponent_arguments: Annotated[List[Transcript], reducer]
    audience_profile: Annotated[AudienceProfile, reducer]
    factual_analysis: Annotated[Feedback, reducer]
    logical_structure: Annotated[Feedback, reducer]
    logical_fallacies: Annotated[Feedback, reducer]
    emotional_appeal: Annotated[Feedback, reducer]
    style_clarity: Annotated[Feedback, reducer]
    complex_feedback: Annotated[Feedback, reducer]
