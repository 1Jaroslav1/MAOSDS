import operator

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Sequence, Annotated, List
from agents.hub import gpt_4o_mini, get_tavily_tool
from langgraph.graph import StateGraph, START

tavily_tool = get_tavily_tool(max_results=5)
tools = [tavily_tool]

class TeamMember(TypedDict):
    name: str
    description: str


class TeamState(TypedDict):
    topic: str
    opposite_team_arguments: List[BaseMessage]
    members: List[TeamMember]
    team_leader_advice: str
    executed_members: List[TeamMember]
    team_arguments: List[str]


def create_team_leader(parent_node: str):
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
                input_variables=["members", "executed_members", "remaining_members", "team_arguments", "opposite_team_arguments"],
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
                "next": parent_node
            }

    return team_leader_node

class TeamMemberNode:
    def __init__(self, member: TeamMember):
        self.member: TeamMember = member

    def get_member_name(self):
        return self.member["name"]

    def create_member_node(self):

        class TeamMemberOutput(BaseModel):
            arguments: str = Field(description="Member arguments that enhances his team's overall position")

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
                input_variables=["member_description", "team_arguments", "team_leader_advice", "opposite_team_arguments"],
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


def is_all_members_executed(state: TeamState):
    if len(state["executed_members"]) == len(state["members"]):
        return

def create_team_graph(topic: str, members: List[TeamMember]) -> StateGraph:
    team_tools_node = ToolNode(tools)

    workflow = StateGraph(TeamState)
    team_leader_node = create_team_leader(END)
    workflow.add_node("team_leader_node", team_leader_node)
    workflow.add_node("team_tools_node", team_tools_node)
    
    for member in members:
        team_member_node = TeamMemberNode(member=member)

        member_node = team_member_node.create_member_node()
        member_name = member["name"]

        workflow.add_node(member_name, member_node)
        workflow.add_edge(member_name, "team_leader_node")

    workflow.add_edge(START, "team_leader_node")
    workflow.add_conditional_edges("team_leader_node", lambda state: state["next"])

    return workflow
