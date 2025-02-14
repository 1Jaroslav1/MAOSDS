from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from agents.hub.llm_hub import gpt_4o_mini
from state import AudienceMember, AudienceState, Decision


class AudienceMemberDecisionOutput(BaseModel):
    decision: str = Field(description="Decision: either 'agree' or 'disagree'.")


def create_init_decision_node(member: AudienceMember):
    def init_decision(state: AudienceState):
        prompt = PromptTemplate(
            template="""
                        You are an audience member participating in a debate. Please read your profile carefully and reflect on how your experiences and personal values shape your views on wage policies.

                        Your personal profile:
                        - Name: {name}
                        - Interests: {interests}
                        - Work Experience: {work_experience}
                        - Personality: {personality}
                        
                        Debate Topic:
                        {topic}
                        
                        Reflect on your background:
                        - Does your work experience or industry make you value merit-based compensation over strict equality?
                        - Do your interests and personal beliefs lean toward an ideal of egalitarianism, regardless of industry differences?
                        
                        Based on your profile and reflection on these considerations, decide whether you personally "agree" or "disagree" with the statement.
                        
                        Provide only a single-word answer: either "agree" or "disagree".
                        """,
            input_variables=["name", "interests", "work_experience", "personality", "topic"]
        )

        chain = prompt | gpt_4o_mini.with_structured_output(AudienceMemberDecisionOutput)
        result = chain.invoke({
            "name": member["name"],
            "interests": member["interests"],
            "work_experience": member["work_experience"],
            "personality": member["personality"],
            "topic": state["topic"]
        })

        decision: Decision = {
            "name": member["name"],
            "value": result.decision
        }

        state["initial_scores"] += [decision]

        return state

    return init_decision


def create_member(step: str, member: AudienceMember):
    if step == "init":
        return create_init_decision_node(member)
