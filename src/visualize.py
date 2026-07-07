"""
Visualisation du graphe LangGraph. Génère une image du schéma de l'agent.
"""

from graph import app

png_bytes = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("Graphe sauvegardé dans graph.png")