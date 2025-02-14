from typing_extensions import TypedDict, List
from enum import Enum


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


class TeamState(TypedDict):
    topic: str
    team_role: TeamRole
    members: List[TeamMember]
    transcript: List[Transcript]
    audience_profile: AudienceProfile
