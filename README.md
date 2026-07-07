# Agent Agentic RAG — Tourisme au Maroc

Projet réalisé dans le cadre de l'évaluation de fin de module *Agentic RAG*
(Master IIBDCC, Prof. RETAL Sara).

## Objectif

Un agent intelligent qui répond à des questions sur le tourisme au Maroc
(Marrakech, Fès, Chefchaouen...) en combinant :
- une base documentaire interne (RAG) construite à partir d'un guide PDF,
- des outils supplémentaires (météo, estimation de budget),
- un raisonnement autonome (l'agent choisit lui-même l'outil à utiliser),
- une mémoire de conversation.

Construit avec **LangGraph**, graphe fait à la main (sans `create_agent`
de LangChain, comme demandé dans la consigne).

## Stack technique

| Composant | Choix | Pourquoi |
|---|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) | Gratuit, rapide, pas de carte bancaire |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) | Gratuit, tourne en local |
| Base vectorielle | `InMemoryVectorStore` (LangChain) | Simple, suffisant pour ce projet |
| Orchestration | LangGraph | Graphe personnalisé (state + mémoire) |

## Structure du projet
```
evaluation/
├── data/
│   └── guide_tourisme_maroc.pdf   # base documentaire (guide touristique)
├── vectorstore.py                 # Etape 1 : prétraitement + vectorisation
├── tools.py                       # Etape 2 : outils (rechercher_guide, obtenir_meteo, estimer_budget)
├── graph.py                       # Etape 3 : graphe LangGraph (state + mémoire)
├── visualize.py                   # Etape 4 : visualisation du graphe
├── evaluate.py                    # Etape 5 : évaluation (10 questions simples + 10 complexes)
├── resultats_evaluation.csv       # généré après evaluate.py
├── graph.png                      # généré après visualize.py
├── requirements.txt
├── .env                           # contient GROQ_API_KEY (non versionné)
└── README.md
```

## Installation

### Avec uv

Le projet utilise `uv` avec `pyproject.toml` (créé via `uv init`, dépendances
ajoutées via `uv add`).

```bash
uv sync
```

Cette commande crée automatiquement l'environnement virtuel (`.venv`) et
installe toutes les dépendances listées dans `pyproject.toml`.

Pour lancer un script :
```bash
uv run python vectorstore.py
```

Pour ajouter une nouvelle dépendance (si besoin) :
```bash
uv add langchain-huggingface
```

### Configuration

Créer un fichier `.env` à la racine avec ta clé Groq (gratuite sur
[console.groq.com](https://console.groq.com)) :
```
HF_TOKEN=hf_ta_cle_ici
GROQ_API_KEY=gsk_ta_cle_ici
```

## Utilisation, étape par étape

### 1. Construire la base vectorielle
```bash
python vectorstore.py
```
Charge `data/guide_tourisme_maroc.pdf`, le découpe en chunks, calcule les
embeddings et vérifie qu'une recherche fonctionne.

### 2. Outils
`tools.py` définit 3 outils utilisables par l'agent :
- `rechercher_guide` : recherche documentaire dans le guide PDF
- `obtenir_meteo` : climat général par ville
- `estimer_budget` : calcul de budget de séjour

### 3. Lancer l'agent
```bash
python graph.py
```
Exécute une conversation de test pour vérifier le fonctionnement du graphe
et de la mémoire.

### 4. Visualiser le graphe
```bash
python visualize.py
```
Génère `graph.png` (schéma de l'architecture, utile pour le rapport).

### 5. Évaluer le système
```bash
python evaluate.py
```
Exécute l'agent sur 10 questions simples + 10 questions complexes, mesure
le temps de réponse et les outils utilisés, sauvegarde tout dans
`resultats_evaluation.csv`.

## Architecture du graphe
START → agent → (appel d'outil nécessaire ?)
├── oui → tools → retour vers agent
└── non → END


- **State** : liste des messages de la conversation (`AgentState`).
- **Mémoire** : `MemorySaver` (checkpointer LangGraph), un `thread_id` par
  conversation.

## Résultats de l'évaluation

*(à compléter après avoir lancé `evaluate.py` avec succès)*

- Temps de réponse moyen (questions simples) : ... s
- Temps de réponse moyen (questions complexes) : ... s
- Taux de réponses correctes / pertinentes : ...
- Détail complet dans `resultats_evaluation.csv`

## Limites connues et pistes d'amélioration

- **Modèle gratuit (Groq free tier)** : quota quotidien limité (500k tokens/jour),
  peut nécessiter d'attendre ou de changer de clé API entre les tests.
- **Tool-calling parfois instable** : certains modèles Llama légers
  appellent des outils non pertinents ou résument mal le résultat obtenu ;
  un modèle plus grand (ex: `llama-3.3-70b-versatile`) ou un prompt système
  plus strict améliore ce comportement.
- **Base documentaire limitée** : un seul PDF de 4 pages ; un corpus plus
  riche améliorerait la couverture et la pertinence des réponses.
- **Mémoire non persistante** : `MemorySaver` garde l'historique en RAM
  uniquement ; une vraie application utiliserait `SqliteSaver` ou
  `PostgresSaver`.
- **Pas de re-ranking** des documents récupérés : la pertinence pourrait
  être améliorée en ajoutant une étape de re-ranking après la recherche
  vectorielle.

## Auteur

Projet individuel réalisé par Mehdi, dans le cadre du Master IIBDCC.