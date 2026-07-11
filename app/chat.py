# Ce module orcherstre la conversation :
# 1. Prend la question de l'utilisateur
# 2. Récupère les chunks pertinents depuis ChromaDB
# 3. Construit le prompt avec le contexte
# 4. Appelle Mistral 7B via Ollama (LOCAL, gratuit)
# 5. Retourne la réponse

from openai import OpenAI
from dotenv import load_dotenv
import os
from .embeddings import rechercher_documents_pertinents

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
# La magie d'Ollama : il expose une API identique à OpenAI sur localhost.
# On utilise le même SDK Python (openai), on change juste :
#   - base_url : pointe vers Ollama au lieu des serveurs d'Anthropic
#   - api_key  : valeur factice (Ollama n'authentifie pas)
#   - model    : nom du modèle Ollama ("mistral") au lieu de "gpt-4o-mini"

client_ollama = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama" # Ollama n'authentifie pas, mais le SDK l'exige
)

NOM_MODELE = os.getenv("OLLAMA_MODEL", "mistral")
#

def construire_prompt_systeme() -> str:
    """
    Retourne le prompt système qui définit le comportement du chatbot.

    Le prompt système est crucial : il "programme" le comportement du LLM.
    Avec Mistral (modèle open source), il faut être un peu plus explicite
    qu'avec GPT-4 sur les instructions de langue et de format.
    """
    return """Tu es un assistant FAQ professionnel et serviable.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT en te basant sur le contexte fourni entre les balises <contexte>.
2. Si la réponse n'est pas dans le contexte, dis exactement : "Je ne trouve pas cette information dans notre documentation. Veuillez contacter notre équipe directement."
3. Tu DOIS répondre en français, même si la question est en anglais.
4. Ne fabrique JAMAIS d'informations. Si tu n'es pas sûr, dis-le clairement.
5. Sois poli et professionnel.
6. Donne des réponses concises et directes.

FORMAT DE RÉPONSE :
- Commence par répondre directement à la question
- Ajoute des détails pratiques si pertinent
- Ne répète pas la question dans ta réponse"""

def repondre_a_question(question: str, historique: list[dict] = None) -> dict:
    """
    Fonction principale : génère une réponse à une question en utilisant le RAG

    Flux complet :
    1. Récuperer les chunks pertinents depuis ChromaDB (Retrieval)
    2. Construire le prompt avec ces chunks comme contexte
    3. Appeler Mistral (Generation)
    4. Retourner la reponse + les sources utilisées

    :param question: La question de l'utilisateur
    :param historique: Liste optionnelle des échanges précédents (pour la mémoire)
                        Format : [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    :return: Dictionnaire avec "reponse" (str), "sources" (list[str]) et "modele" (str)
    """

    if historique is None:
        historique = []

    # ÉTAPE 1 : Récupérer les chunks pertinents (Retrieval avec BGE-M3 + ChromaDB)
    chunks_pertinents = rechercher_documents_pertinents(question, n_resultats=3)

    # ÉTAPE 2 : Construire le contexte depuis les chunks
    # On joint les chunks avec des séparateurs clairs pour aider Mistral
    # à distinguer les différentes section du contexte
    contexte = "\n\n---\n\n".join(chunks_pertinents)

    # ÉTAPE 3 : Construire les messages pour Mistral
    messages = [
        # Message système : définit le comportement global
        {"role": "system", "content": construire_prompt_systeme()},
    ]

    # Ajouter l'historique de conversation
    # Cela permet à Mistral de comprendre les questions de suivi :
    # Ex : user "et le samedi ?" après avoir parlé des horaires
    messages.extend(historique)

    # Ajouter la question actuelle AVEC le contexte injecté
    # Note : le contexte est dans le message user, pas dans le system.
    # C'est une pratique recommandée avec les modèles open source.
    messages.append({
        "role": "user",
        "content": f"""Voici les informations disponibles :

<contexte>
{contexte}
</contexte>

Ma question : {question}"""
    })

    # ÉTAPE 4 : Appeler Mistral via Ollama
    # La syntaxe est IDENTIQUE à un appel OpenAI standard
    reponse_api = client_ollama.chat.completions.create(
        model=NOM_MODELE,
        messages=messages,
        temperature=0.1,    # Très bas = réponses factuelles et cohérentes
                            # 0 = toujours la même réponse (déterministe)
                            # 1 = créatif et varié (pas adapté pour FAQ)
        # Note : pas de max_tokens ici car Mistral gère bien la longueur
        # avec un bon prompt. Tu peux l'ajouter si les réponses sont trop longues.
    )

    # Extraire le texte de la réponse (même structure qu'OpenAI)
    texte_reponse = reponse_api.choices[0].message.content

    return  {
        "reponse": texte_reponse,
        "sources": chunks_pertinents,  # Les chunks utilisés (transparence)
        "modele": NOM_MODELE,  # Nom du modèle utilisé
        "tokens_utilises": getattr(reponse_api.usage, "total_tokens", 0)
        # getattr avec default 0 car Ollama ne retourne pas toujours les tokens
    }