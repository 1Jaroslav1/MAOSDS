from langchain_core.prompts import PromptTemplate
from agents.team.team_memeber.state import TeamMemberState
from agents.hub import gpt_4o_mini
from pydantic import BaseModel, Field


class ArgumentationStrategyOutput(BaseModel):
    rhetorical_strategy: str = Field(description="The chosen rhetorical strategy (logos, pathos, ethos) for engaging the audience emotionally and ethically.")
    logical_structure: str = Field(description="The selected logical structure (deductive, inductive, or mixed) for coherent reasoning.")
    factual_strategy: str = Field(description="The approach to integrate and validate factual evidence in the argument.")
    counterargument_strategy: str = Field(description="The strategy for anticipating and refuting potential counterarguments.")
    contextual_adaptation: str = Field(description="The method used to tailor the argument to the specific audience and situational context.")


class CompleteArgumentationOutput(BaseModel):
    argument_draft: str = Field(description="The final drafted argument that integrates all strategies.")
    rhetorical_strategy: str = Field(description="The chosen rhetorical strategy (e.g., logos, pathos, ethos) to emotionally and ethically engage the audience.")
    logical_structure: str = Field(description="The selected logical structure (deductive, inductive, or mixed) to ensure coherent reasoning.")
    factual_strategy: str = Field(description="The approach to integrate and verify factual evidence supporting the argument.")
    counterargument_strategy: str = Field(description="The plan to anticipate and refute potential counterarguments.")
    contextual_adaptation: str = Field(description="The method of tailoring the argument to the specific context and nuances of the audience.")


def argumentation_node(state: TeamMemberState) -> TeamMemberState:
    strategy_prompt = PromptTemplate(
        template="""
                Role:
                You are an expert Argumentation Strategist tasked with designing a comprehensive strategy for constructing a persuasive argument. Your role is to determine the optimal approaches across multiple dimensions of argumentation.
                You should behave like you are human with such personality and experience: {person}
                You are in {team_role} team.
                
                Context:
                In our multi-agent system, it is crucial that each argument is not only persuasive but also robust and resilient to challenges. Therefore, you need to consider:
                - How to emotionally and ethically engage the audience (Rhetorical Strategy).
                - How to structure the reasoning in a clear and coherent manner (Logical Structure).
                - How to incorporate and validate factual evidence (Factual Strategy).
                - How to anticipate potential objections and refute them (Counterargument Strategy).
                - How to adapt the argument to the specific audience and situational context (Contextual Adaptation).
                
                Additional Instructions:
                - This is a reprocessing cycle based on evaluator feedback. Please review the following evaluation details:
                  Evaluation Summary: {evaluation_summary}
                  Suggestions: {evaluation_suggestions}
                - Revise your strategy accordingly to address any identified issues (e.g., logical fallacies, insufficient evidence, or unclear structure).

                Inputs:
                - Topic: {topic}
                - Analysis Summary: {analysis_summary}
                - Retrieved Evidence: {evidence_summary}
                - Audience Profile: {audience_profile}

                Tasks:
                Based on the above inputs, determine the best strategies for each dimension. 

                Please output your result in JSON format with the following keys:
                - "rhetorical_strategy": (logos, pathos, ethos)
                - "logical_structure": (deductive, inductive, or mixed)
                - "factual_strategy": (how to integrate and validate factual evidence)
                - "counterargument_strategy": (how to anticipate and refute potential counterarguments)
                - "contextual_adaptation": (how to tailor the argument to the audience)
            """,
        input_variables=["topic", "team_role", "analysis_summary", "evidence_summary", "audience_profile", "evaluation_summary", "evaluation_suggestions"]
    )
    strategy_chain = strategy_prompt | gpt_4o_mini.with_structured_output(ArgumentationStrategyOutput)
    strategy_result = strategy_chain.invoke({
        "topic": state["topic"],
        "team_role": state["team_role"],
        "person": state["person"],
        "analysis_summary": state["analysis"]["main_themes_and_issues"],
        "evidence_summary": state["retrieved_data"]["evidence_summary"],
        "audience_profile": state["audience_profile"],
        "evaluation_summary": state["evaluation"].get("evaluation_summary", "No previous evaluation available."),
        "evaluation_suggestions": state["evaluation"].get("suggestions", "No suggestions provided.")
    })

    combined_analysis = state["analysis"]["main_themes_and_issues"] + state.get("team_arguments", [])

    argument_prompt = PromptTemplate(
        template="""
            Role:
            You are the Argumentation Agent responsible for drafting a persuasive and comprehensive argument that integrates multiple reasoning strategies.
            You should create arguments like you are human with such personality and experience: {person}
            
            Context:
            Your argument should be robust by ensuring:
            - Verified factual evidence is seamlessly integrated (Factual Strategy).
            - The logical structure is clear and coherent (Logical Structure).
            - The rhetorical approach resonates with the audience (Rhetorical Strategy).
            - Potential counterarguments are anticipated and addressed (Counterargument Strategy).
            - The argument is adapted to the specific context and audience (Contextual Adaptation).

            Inputs:
            - Topic: {topic}
            - Analysis Summary and Team Context: {analysis_summary}
            - Retrieved Evidence: {evidence_summary}
            - Audience Profile: {audience_profile}
            - Chosen Rhetorical Strategy: {rhetorical_strategy}
            - Chosen Logical Structure: {logical_structure}
            - Chosen Factual Strategy: {factual_strategy}
            - Chosen Counterargument Strategy: {counterargument_strategy}
            - Chosen Contextual Adaptation: {contextual_adaptation}

            Tasks:
            Craft a complete and persuasive argument that weaves together the above strategies. 
            Provide your output in JSON format with the keys:
            "argument_draft", "rhetorical_strategy", "logical_structure", "factual_strategy", "counterargument_strategy", "contextual_adaptation".
        """,
        input_variables=[
            "topic", "person",
            "analysis_summary", "evidence_summary", "audience_profile",
            "rhetorical_strategy", "logical_structure", "factual_strategy",
            "counterargument_strategy", "contextual_adaptation"
        ]
    )
    argument_chain = argument_prompt | gpt_4o_mini.with_structured_output(CompleteArgumentationOutput)
    argument_result = argument_chain.invoke({
        "topic": state["topic"],
        "person": state["person"],
        "analysis_summary": combined_analysis,
        "evidence_summary": state["retrieved_data"]["evidence_summary"],
        "audience_profile": state["audience_profile"],
        "rhetorical_strategy": getattr(strategy_result, "rhetorical_strategy", "logos"),
        "logical_structure": getattr(strategy_result, "logical_structure", "deductive"),
        "factual_strategy": getattr(strategy_result, "factual_strategy", "verify and integrate data from credible sources"),
        "counterargument_strategy": getattr(strategy_result, "counterargument_strategy", "anticipate objections and provide rebuttals"),
        "contextual_adaptation": getattr(strategy_result, "contextual_adaptation", "tailor argument based on audience profile"),
    })

    state["argument"] = {
        "argument_draft": argument_result.argument_draft,
        "rhetorical_strategy": argument_result.rhetorical_strategy,
        "logical_structure": argument_result.logical_structure,
        "factual_strategy": argument_result.factual_strategy,
        "counterargument_strategy": argument_result.counterargument_strategy,
        "contextual_adaptation": argument_result.contextual_adaptation,
    }

    return state
