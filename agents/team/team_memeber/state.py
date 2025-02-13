from enum import Enum
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, List, Optional


class TeamRole(str, Enum):
    PROPOSING = 'PROPOSING'
    OPPOSING = 'OPPOSING'
    CHAIRMAN = 'CHAIRMAN'
    AUDIENCE = 'AUDIENCE'


class TeamMember(TypedDict):
    name: str
    expertise: str
    description: str


class Transcript(TypedDict):
    speaker: TeamMember
    team_role: TeamRole
    text: str


class AudienceProfile:
    demographics: str
    interests: List[str]


class AnalysisState(TypedDict):
    main_themes_and_issues: List[str]
    opponent_perspectives: List[str]
    opponent_weaknesses: List[str]


class Evidence(TypedDict):
    summary: str
    source: str
    url: str
    publication_date: Optional[str]


class RetrievedData(TypedDict):
    evidence_summary: str
    evidence_items: List[Evidence]


class Argument(TypedDict):
    argument_draft: str
    rhetorical_strategy: str
    logical_structure: str
    factual_strategy: str
    counterargument_strategy: str
    contextual_adaptation: str


class LexiconAdjustment(TypedDict):
    refined_argument: str
    refinement_notes: str


class Evaluation(TypedDict):
    evaluation_summary: str
    style_evaluation: str
    logical_evaluation: str
    suggestions: str
    reprocess: bool


class TeamMemberState(TypedDict):
    topic: str
    team_role: TeamRole
    person: TeamMember
    transcript: List[Transcript]
    team_arguments: List[str]
    opponent_arguments: List[str]
    audience_profile: AudienceProfile
    analysis: AnalysisState
    retrieved_data: RetrievedData
    argument: Argument
    lexicon_adjustment: LexiconAdjustment
    evaluation: Evaluation
    iteration_number: int


class TeamState(TypedDict):
    topic: str
    team_arguments: List[str]
    members: List[TeamMember]
    team_role: TeamRole
    transcript: List[Transcript]
    opponent_arguments: List[str]
    audience_profile: AudienceProfile
    analysis: AnalysisState
    retrieved_data: RetrievedData
    argument: Argument
    lexicon_adjustment: LexiconAdjustment
    evaluation: Evaluation
    iteration_number: int
