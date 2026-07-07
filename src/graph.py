"""
Architecture Agentic RAG avec LangGraph — CONSTRUITE A LA MAIN, sans
create_agent (interdit par la consigne car il fournit une boucle de
raisonnement déjà prête). On recrée nous-mêmes cette boucle :
LLM réfléchit -> appelle un outil si besoin -> lit le résultat -> répond.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from tools import all_tools

load_dotenv()


# --- 1. STATE : ce qui circule dans le graphe -------------------------------
class AgentState(TypedDict):
    # add_messages empile les messages au lieu de les remplacer : c'est ce
    # qui donne l'historique de conversation.
    messages: Annotated[list, add_messages]


# --- 2. LE LLM ---------------------------------------------------------------
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
model_with_tools = model.bind_tools(all_tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "Tu es un agent utile qui aide les utilisateurs en cherchant des "
    "informations dans le guide touristique du Maroc. Utilise UN SEUL outil "
    "à la fois, celui qui correspond exactement à la question posée : "
    "rechercher_guide pour les infos du guide, obtenir_meteo UNIQUEMENT "
    "pour la météo, estimer_budget UNIQUEMENT si un budget est demandé. "
    "IMPORTANT : après avoir reçu le résultat d'un outil, rédige une réponse "
    "COMPLÈTE en français qui reprend concrètement les informations retournées. "
    "Ne dis JAMAIS des phrases comme 'la réponse est basée sur la fonction X' "
    "— donne directement le contenu utile à l'utilisateur. "
    "Si tu ne trouves pas l'information, dis-le clairement."
))


# --- 3. LES NODES -------------------------------------------------------------
def agent_node(state: AgentState):
    """Le LLM lit l'historique et décide : répondre ou appeler un outil."""
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tools_node = ToolNode(all_tools)


def should_continue(state: AgentState):
    """Arête conditionnelle : dernier message a-t-il un tool_call ?"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# --- 4. CONSTRUCTION DU GRAPHE -------------------------------------------------
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tools_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # après l'outil, on repasse par l'agent

# --- 5. MEMOIRE ----------------------------------------------------------------
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo"}}
    question = "Quels sont les boissons et pâtisseries incontournables au Maroc ?"
    result = app.invoke({"messages": [("user", question)]}, config=config)
    print(result["messages"][-1].content)