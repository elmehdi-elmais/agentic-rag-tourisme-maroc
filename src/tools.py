"""
Outils (tools) disponibles pour l'agent. Un vrai agent "Agentic RAG" ne fait
pas QUE de la recherche documentaire : il choisit l'outil adapté selon la
question. Ici on a 3 outils : recherche documentaire, météo, budget.
"""

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from vectorstore import build_vectorstore

# La base vectorielle est construite une seule fois au chargement du module,
# puis réutilisée par tous les outils qui en ont besoin.
vector_store: InMemoryVectorStore = build_vectorstore()


@tool
def rechercher_guide(query: str) -> str:
    """Recherche dans le guide touristique du Maroc les passages pertinents
    pour répondre à la question. A utiliser en priorité pour toute question
    sur les villes, sites, gastronomie ou culture du Maroc."""
    results = vector_store.similarity_search(query, k=5)
    if not results:
        return "Aucune information trouvée dans le guide."
    # On concatène les 3 meilleurs extraits (au lieu d'un seul comme dans la
    # version initiale) pour donner plus de contexte au LLM.
    return "\n\n---\n\n".join(r.page_content for r in results)


@tool
def obtenir_meteo(ville: str) -> str:
    """Donne une indication météo générale (climat moyen) pour une ville
    touristique marocaine. A utiliser pour les questions de météo/climat."""
    climats = {
        "marrakech": "Étés très chauds (>38°C), hivers doux (~18°C). Idéal : mars-mai, septembre-novembre.",
        "chefchaouen": "Climat de montagne, étés doux, hivers frais et pluvieux. Idéal : avril-mai, septembre-octobre.",
        "fes": "Étés chauds et secs, hivers frais. Idéal : mars-mai, septembre-novembre.",
        "fès": "Étés chauds et secs, hivers frais. Idéal : mars-mai, septembre-novembre.",
    }
    return climats.get(
        ville.strip().lower(),
        f"Pas de données météo précises pour {ville}, mais le Maroc est agréable au printemps et en automne.",
    )


@tool
def estimer_budget(nb_jours: int, budget_jour_dh: int = 500) -> str:
    """Estime le budget total approximatif d'un séjour au Maroc, à partir
    du nombre de jours et d'un budget journalier moyen en dirhams (DH)."""
    total = nb_jours * budget_jour_dh
    return f"Budget estimé pour {nb_jours} jour(s) à {budget_jour_dh} DH/jour : environ {total} DH (hors avion)."


all_tools = [rechercher_guide, obtenir_meteo, estimer_budget]