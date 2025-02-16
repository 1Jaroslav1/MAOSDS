from langgraph.graph import StateGraph, START, END
from agents.tutor.emotional_appeal_node import emotional_feedback_node
from agents.tutor.factual_accuracy_node import factual_accuracy_node
from agents.tutor.logical_structure_node import logical_structure_feedback_node
from agents.tutor.logical_fallacies_node import logical_fallacies_feedback_node
from agents.tutor.style_clarity_node import style_clarity_feedback_node
from agents.tutor.tutur_summarize import summary_feedback_node
from agents.tutor.state import TutorState


def creat_tutor():
    workflow = StateGraph(TutorState)

    workflow.add_node("logical_structure_feedback_node", logical_structure_feedback_node)
    workflow.add_node("logical_fallacies_feedback_node", logical_fallacies_feedback_node)
    workflow.add_node("factual_accuracy_node", factual_accuracy_node)
    workflow.add_node("emotional_feedback_node", emotional_feedback_node)
    workflow.add_node("style_clarity_feedback_node", style_clarity_feedback_node)
    workflow.add_node("complex_feedback_node", summary_feedback_node)

    workflow.add_edge(START, "logical_structure_feedback_node")
    workflow.add_edge(START, "logical_fallacies_feedback_node")
    workflow.add_edge(START, "factual_accuracy_node")
    workflow.add_edge(START, "emotional_feedback_node")
    workflow.add_edge(START, "style_clarity_feedback_node")
    workflow.add_edge("logical_structure_feedback_node", "complex_feedback_node")
    workflow.add_edge("logical_fallacies_feedback_node", "complex_feedback_node")
    workflow.add_edge("factual_accuracy_node", "complex_feedback_node")
    workflow.add_edge("emotional_feedback_node", "complex_feedback_node")
    workflow.add_edge("style_clarity_feedback_node", "complex_feedback_node")
    workflow.add_edge("complex_feedback_node", END)

    return workflow
