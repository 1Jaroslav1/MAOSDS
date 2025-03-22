from typing_extensions import TypedDict, List
from agents.model.model import TeamRole, Transcript, AudienceProfile


class TeamState(TypedDict):
    topic: str
    team_role: TeamRole
    transcript: List[Transcript]
    audience_profile: AudienceProfile
