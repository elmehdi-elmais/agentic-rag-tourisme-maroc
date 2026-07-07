"""
Evaluation du système sur 10 questions simples + 10 complexes.
Mesure : temps de réponse, outils utilisés, réponse générée.
"""

import time
import csv
from graph import app
from langgraph.errors import GraphRecursionError

questions_simples = [
    "Quelle est la météo générale à marrakech ?",
    "Quels sont les incontournables à voir à Marrakech ?",
    "Qu'est-ce que la place Jemaa el-Fna ?",
    "Quelle est la meilleure période pour visiter Chefchaouen ?",
    "Pourquoi Chefchaouen est-elle appelée la ville bleue ?",
    "Quels sont les plats typiques de Fès ?",
    "Comment se déplacer dans la médina de Fès ?",
    "Quel est l'aéroport le plus proche de Marrakech ?",
    "Qu'est-ce que l'université Al Quaraouiyine ?",
    "Quel budget pour un séjour de 5 jours à 400 DH par jour ?",
]

questions_complexes = [
    "Je pars 7 jours à Marrakech, donne-moi les sites à visiter, la météo et un budget.",
    "Compare Marrakech et Fès en gastronomie et sites historiques.",
    "Itinéraire Chefchaouen + Fès : que voir dans chaque ville ?",
    "Été et je n'aime pas la chaleur : Marrakech, Fès ou Chefchaouen ?",
    "Budget pour 10 jours à Fès, puis les 3 sites historiques à voir.",
    "Différences entre les médinas de Marrakech et Fès ?",
    "Où manger la meilleure pastilla et pourquoi ?",
    "Randonnée et culture ensemble : quelle ville et quelles activités ?",
    "Plan de 3 jours à Chefchaouen avec météo et budget à 350 DH/jour.",
    "Origine de la couleur bleue à Chefchaouen, lien religieux ?",
]


def run(questions, categorie, prefix):
    resultats = []
    for i, q in enumerate(questions, 1):
        config = {"configurable": {"thread_id": f"{prefix}_{i}"}, "recursion_limit": 15}
        debut = time.time()

        try:
            reponse = app.invoke({"messages": [("user", q)]}, config=config)
            texte = reponse["messages"][-1].content
            outils = [m.name for m in reponse["messages"] if getattr(m, "name", None)]
        except GraphRecursionError:
            texte = "ERREUR: boucle infinie détectée"
            outils = []
        except Exception as e:
            texte = f"ERREUR: {e}"
            outils = []

        duree = round(time.time() - debut, 2)

        resultats.append({
            "categorie": categorie,
            "question": q,
            "reponse": texte,
            "temps_secondes": duree,
            "outils_utilises": ", ".join(outils) if outils else "aucun",
        })

        print(f"\n[{categorie}] Q{i}/10 - {duree}s")
        print(f"  Question : {q}")
        print(f"  Outils   : {', '.join(outils) if outils else 'aucun'}")
        print(f"  Réponse  : {texte[:200]}{'...' if len(texte) > 200 else ''}")

        time.sleep(2)  # évite de re-déclencher le rate limit Groq
    return resultats


if __name__ == "__main__":
    res = run(questions_simples, "simple", "simple") + run(questions_complexes, "complexe", "complexe")

    with open("resultats_evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["categorie", "question", "reponse", "temps_secondes", "outils_utilises"])
        writer.writeheader()
        writer.writerows(res)

    print("\nRésultats sauvegardés dans resultats_evaluation.csv")