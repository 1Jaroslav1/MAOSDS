from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from .model import TeamState, TeamMember


class TeamMemberOutput(BaseModel):
    arguments: str = Field(description="Member arguments that enhances his team's overall position")


class TeamMemberNode:
    def __init__(self, member: TeamMember):
        self.member: TeamMember = member

    def get_member_name(self):
        return self.member["name"]

    def create_member_node(self):
        def member_node(state: TeamState):
            team_leader_prompt = PromptTemplate(
                template="""
                    {member_description}

                    **Your Responsibilities:**
                    1. **Analyze the Debate State:** Carefully evaluate the opposing team's arguments ({opposite_team_arguments}) and consider the strategic advice provided by the team leader ({team_leader_advice}).
                    2. **Contribute to Team Success:** Develop a strong, well-crafted argument that enhances your team's overall position ({team_arguments}). Ensure your contribution either addresses weaknesses in the opposing team's arguments or reinforces your team's core narrative.

                    **Current Status:**
                    - **Your Team's Arguments:** {team_arguments}.
                    - **Team Leader's Advice:** {team_leader_advice}.
                    - **Opposing Team's Arguments:** {opposite_team_arguments}.

                    **Objective:** Your goal is to strengthen your team's stance by creating a compelling argument that adds value to the debate. Focus on addressing critical gaps, providing rebuttals, or introducing innovative perspectives.
                """,
                input_variables=["member_description", "team_arguments", "team_leader_advice",
                                 "opposite_team_arguments"],
            )
            member_chain = team_leader_prompt | gpt_4o_mini.with_structured_output(TeamMemberOutput)
            result = member_chain.invoke({
                "member_description": self.member["description"],
                "team_arguments": state["team_arguments"],
                "team_leader_advice": state["team_leader_advice"],
                "opposite_team_arguments": state["opposite_team_arguments"]
            })
            return {
                "team_arguments": state["team_arguments"] + [result.arguments],
                "executed_members": state["executed_members"] + [self.member],
            }

        return member_node
