from agents import TeamMember, TeamState
from typing_extensions import TypedDict, List, Dict


class Team(TypedDict):
    name: str
    topic: str
    members: List[TeamMember]


class DebateState(TypedDict):
    teams: List[Team]
    team_arguments: Dict[str, List[str]]
    executed_teams: List[Team]
    next_team_state: TeamState
    next: str
