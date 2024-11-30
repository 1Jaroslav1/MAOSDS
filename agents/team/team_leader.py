from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from pydantic import BaseModel, Field
from agents.hub import gpt_4o_mini
from .model import TeamState


class TeamLeaderMemberOutput(BaseModel):
    next_person: str = Field(description="The most suitable member to address the identified gaps or to present a strong, new argument")
    advice_for_next_person: str = Field(description="Advice or guidance to help the selected member craft a highly impactful argument that aligns with the team's strategy")


def team_leader_node(state: TeamState):
    members = state['members']
    executed_members = state["executed_members"]
    remaining_members = [m for m in members if m not in executed_members]

    if remaining_members:
        # next_member = random.choice(remaining_members)
        team_leader_prompt = PromptTemplate(
            template="""
                You are a proactive and strategic team leader responsible for managing a high-performing debate team. Your team comprises the following members:
                {members}. Each member is tasked with delivering a unique and impactful argument.

                **Your Responsibilities:**
                1. **Analyze Opposing Arguments:** Review the arguments presented by the opposing team ({opposite_team_arguments}). Identify any gaps or weaknesses that require further discussion.
                2. **Select the Next Member:** Choose the most suitable member ({remaining_members}) to address the identified gaps or to present a strong, new argument. Ensure your choice aligns with the member's expertise and the debate strategy.
                3. **Guide Argumentation:** Provide clear guidance to the selected member, ensuring their contribution effectively counters the opposing team's arguments or strengthens your team's overall position.
                4. **Offer Strategic Support:** Suggest specific directions, evidence, or rebuttals to help the selected member craft a compelling and strategic argument.

                **Current Debate Status:**
                - **Your Team's Arguments:** {team_arguments}.
                - **Members Who Have Already Presented:** {executed_members}.
                - **Remaining Members:** {remaining_members}.
                - **Opposing Team's Arguments:** {opposite_team_arguments}.

                **Objective:** Your goal is to lead your team to debate victory by ensuring each member's contribution adds maximum value. This can include defending unresolved counterarguments, presenting innovative ideas, or delivering strategic insights to outmaneuver the opposing team.
            """,
            input_variables=["members", "executed_members", "remaining_members", "team_arguments",
                             "opposite_team_arguments"],
        )

        llm_chain = team_leader_prompt | gpt_4o_mini.with_structured_output(TeamLeaderMemberOutput)
        result = llm_chain.invoke(
            {
                "members": members,
                "executed_members": executed_members,
                "remaining_members": remaining_members,
                "team_arguments": state["team_arguments"],
                "opposite_team_arguments": state["opposite_team_arguments"]
            }
        )
        return {
            "team_leader_advice": result.advice_for_next_person,
            "next": result.next_person
        }
    else:
        return {
            "team_arguments": state["team_arguments"],
            "next": END
        }
