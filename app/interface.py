"""
Interface Streamlit — Chatbot FAQ
Projet 1 : RAG avec FastAPI + ChromaDB + BGE-M3 + Mistral

Lancement :
    streamlit run interface.py

L'interface se connecte au backend FastAPI sur http://localhost:8000
Lance d'abord : python -m uvicorn app.main:app --reload
"""

import streamlit as st
import requests
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Assistant FAQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Police et fond général */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fond de la page */
.stApp {
    background-color: #0f1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2a2d3a;
}

/* Bulles de chat — utilisateur */
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 60px;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
}

/* Bulles de chat — assistant */
.bubble-assistant {
    background-color: #1e2130;
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 60px 6px 0;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid #2a2d3a;
}

/* Zone sources */
.sources-box {
    background-color: #131620;
    border: 1px solid #2a2d3a;
    border-left: 3px solid #4f46e5;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 13px;
    color: #94a3b8;
}

/* Label rôle */
.role-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.role-label-user {
    color: #818cf8;
    text-align: right;
}

.role-label-assistant {
    color: #34d399;
}

/* Badge statut */
.badge-online {
    display: inline-block;
    background-color: #064e3b;
    color: #34d399;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #065f46;
}

.badge-offline {
    display: inline-block;
    background-color: #450a0a;
    color: #f87171;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #7f1d1d;
}

/* Titre principal */
.main-title {
    font-size: 28px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 24px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2a2d3a;
    margin: 16px 0;
}

/* Timestamp */
.timestamp {
    font-size: 11px;
    color: #475569;
    margin-top: 3px;
}

/* Section titre sidebar */
.sidebar-section {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin: 16px 0 8px;
}

/* Stats box */
.stat-box {
    background-color: #131620;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #818cf8;
}

.stat-label {
    font-size: 12px;
    color: #64748b;
}

/* Suggestion chips */
.suggestion {
    display: inline-block;
    background-color: #1e2130;
    border: 1px solid #2a2d3a;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #94a3b8;
    cursor: pointer;
    margin: 3px;
    transition: all 0.2s;
}

.suggestion:hover {
    border-color: #4f46e5;
    color: #818cf8;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — Mémoire de l'interface entre les interactions
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # Historique des messages affichés

if "historique_api" not in st.session_state:
    st.session_state.historique_api = []    # Historique formaté pour l'API

if "document_indexe" not in st.session_state:
    st.session_state.document_indexe = None # Nom du document chargé

if "nb_questions" not in st.session_state:
    st.session_state.nb_questions = 0       # Compteur de questions posées

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def verifier_api() -> bool:
    """Vérifie que le backend FastAPI est accessible."""
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def uploader_document(fichier) -> dict:
    """
    Envoie un fichier au backend pour indexation.
    Retourne la réponse JSON du backend.
    """
    try:
        reponse = requests.post(
            f"{API_URL}/document/upload",
            files={"fichier": (fichier.name, fichier.getvalue(), fichier.type)},
            timeout=120,  # L'indexation peut prendre du temps (génération embeddings)
        )
        return reponse.json()
    except requests.exceptions.Timeout:
        return {"erreur": "Timeout — l'indexation prend trop de temps. Réessaye."}
    except Exception as e:
        return {"erreur": str(e)}


def poser_question(question: str, historique: list) -> dict:
    """
    Envoie une question au backend et retourne la réponse.
    """
    try:
        reponse = requests.post(
            f"{API_URL}/chat",
            json={
                "question": question,
                "historique": historique,
            },
            timeout=60,  # Mistral peut prendre jusqu'à 30-40s sur première requête
        )
        return reponse.json()
    except requests.exceptions.Timeout:
        return {"erreur": "Mistral met trop de temps à répondre. Réessaye dans quelques secondes."}
    except Exception as e:
        return {"erreur": str(e)}


def afficher_message(role: str, contenu: str, sources: list = None, timestamp: str = None):
    """
    Affiche une bulle de message dans l'interface.

    Args:
        role: "user" ou "assistant"
        contenu: Le texte du message
        sources: Les chunks sources (uniquement pour l'assistant)
        timestamp: L'heure du message
    """
    if role == "user":
        st.markdown(f'<div class="role-label role-label-user">Vous</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-user">{contenu}</div>', unsafe_allow_html=True)
        if timestamp:
            st.markdown(f'<div class="timestamp" style="text-align:right">{timestamp}</div>', unsafe_allow_html=True)

    else:  # assistant
        st.markdown(f'<div class="role-label role-label-assistant">🤖 Assistant</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-assistant">{contenu}</div>', unsafe_allow_html=True)
        if timestamp:
            st.markdown(f'<div class="timestamp">{timestamp}</div>', unsafe_allow_html=True)

        # Afficher les sources si disponibles
        if sources:
            with st.expander("📎 Sources utilisées", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="sources-box">
                        <strong style="color:#818cf8">Source {i}</strong><br>
                        {source[:300]}{"..." if len(source) > 300 else ""}
                    </div>
                    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    # Logo et titre
    st.markdown("## 🤖 Assistant FAQ")
    st.markdown("*Propulsé par Mistral + RAG*")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Statut de l'API
    st.markdown('<div class="sidebar-section">Statut</div>', unsafe_allow_html=True)
    api_ok = verifier_api()
    if api_ok:
        st.markdown('<span class="badge-online">● API connectée</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">● API hors ligne</span>', unsafe_allow_html=True)
        st.warning("Lance le backend :\n```\npython -m uvicorn app.main:app --reload\n```")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Upload de document
    st.markdown('<div class="sidebar-section">Document</div>', unsafe_allow_html=True)

    fichier = st.file_uploader(
        "Charger un document",
        type=["txt", "pdf"],
        help="Le document sera découpé et indexé dans ChromaDB",
        label_visibility="collapsed",
    )

    if fichier:
        if st.button("📥 Indexer le document", use_container_width=True, type="primary"):
            with st.spinner(f"Indexation de '{fichier.name}'..."):
                resultat = uploader_document(fichier)

            if "erreur" in resultat:
                st.error(f"Erreur : {resultat['erreur']}")
            else:
                st.session_state.document_indexe = fichier.name
                st.success(f"✅ {resultat.get('nombre_chunks', '?')} chunks indexés !")

    # Afficher le document chargé
    if st.session_state.document_indexe:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Document actif</div>
            <div style="color:#e2e8f0; font-size:13px; margin-top:4px; font-weight:500">
                📄 {st.session_state.document_indexe}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Statistiques
    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{st.session_state.nb_questions}</div>
            <div class="stat-label">Questions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{len(st.session_state.messages) // 2}</div>
            <div class="stat-label">Échanges</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Bouton reset
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.historique_api = []
        st.session_state.nb_questions = 0
        st.rerun()

    # Infos modèle
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#475569; line-height:1.8">
        🧠 <b style="color:#64748b">LLM</b> : Mistral 7B (Ollama)<br>
        📊 <b style="color:#64748b">Embeddings</b> : BGE-M3<br>
        🗄️ <b style="color:#64748b">Vector DB</b> : ChromaDB<br>
        ⚡ <b style="color:#64748b">Backend</b> : FastAPI
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — Chat
# ─────────────────────────────────────────────────────────────────────────────

# En-tête
st.markdown('<div class="main-title">Assistant FAQ Intelligent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Posez vos questions sur le document chargé — '
    'réponses précises basées uniquement sur votre contenu</div>',
    unsafe_allow_html=True
)

# Message de bienvenue si aucun message
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px; color: #475569">
        <div style="font-size: 48px; margin-bottom: 16px">💬</div>
        <div style="font-size: 18px; font-weight: 600; color: #64748b; margin-bottom: 8px">
            Commencez par charger un document
        </div>
        <div style="font-size: 14px">
            Glissez un fichier PDF ou TXT dans la barre latérale,<br>
            puis posez vos questions ici.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Questions suggérées
    if st.session_state.document_indexe:
        st.markdown("**Questions suggérées :**")
        suggestions = [
            "Quels sont les horaires ?",
            "Comment s'inscrire ?",
            "Quels sont les tarifs ?",
            "Qui contacter ?",
        ]
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["question_suggeree"] = suggestion
                    st.rerun()

# Afficher l'historique des messages
for msg in st.session_state.messages:
    afficher_message(
        role=msg["role"],
        contenu=msg["contenu"],
        sources=msg.get("sources"),
        timestamp=msg.get("timestamp"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# ZONE DE SAISIE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Récupérer une question suggérée si l'utilisateur a cliqué sur un bouton
valeur_defaut = st.session_state.pop("question_suggeree", "")

col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "question",
        value=valeur_defaut,
        placeholder="Posez votre question ici...",
        label_visibility="collapsed",
        key="input_question",
    )

with col_btn:
    envoyer = st.button("Envoyer ➤", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAITEMENT DE LA QUESTION
# ─────────────────────────────────────────────────────────────────────────────

if (envoyer or valeur_defaut) and question.strip():

    # Vérifications préalables
    if not api_ok:
        st.error("❌ Le backend FastAPI n'est pas accessible. Lance `python -m uvicorn app.main:app --reload`")
        st.stop()

    if not st.session_state.document_indexe:
        st.warning("⚠️ Charge d'abord un document dans la barre latérale.")
        st.stop()

    # Ajouter le message utilisateur à l'historique affiché
    timestamp_now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "contenu": question,
        "timestamp": timestamp_now,
    })
    st.session_state.nb_questions += 1

    # Appeler l'API avec un spinner
    with st.spinner("🤖 Mistral réfléchit..."):
        resultat = poser_question(
            question=question,
            historique=st.session_state.historique_api,
        )

    # Traiter la réponse
    if "erreur" in resultat:
        reponse_texte = f"❌ Erreur : {resultat['erreur']}"
        sources = []
    else:
        reponse_texte = resultat.get("reponse", "Aucune réponse reçue.")
        sources = resultat.get("sources", [])

        # Mettre à jour l'historique API pour la prochaine question
        st.session_state.historique_api.append({
            "role": "user",
            "content": question,
        })
        st.session_state.historique_api.append({
            "role": "assistant",
            "content": reponse_texte,
        })

        # Limiter l'historique à 6 derniers échanges (évite de dépasser le contexte)
        if len(st.session_state.historique_api) > 12:
            st.session_state.historique_api = st.session_state.historique_api[-12:]

    # Ajouter la réponse à l'historique affiché
    st.session_state.messages.append({
        "role": "assistant",
        "contenu": reponse_texte,
        "sources": sources,
        "timestamp": datetime.now().strftime("%H:%M"),
    })

    # Recharger la page pour afficher les nouveaux messages
    st.rerun()
