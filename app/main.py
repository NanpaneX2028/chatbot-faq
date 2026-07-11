import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from pathlib import Path
from .ingestion import charger_document
from .embeddings import indexer_documents
from .chat import repondre_a_question

load_dotenv()

# Créer l'application FastAPI avec des métadonnées
app = FastAPI(
    title="Chatbot FAQ",
    description="Chatbot FAQ avec RAG (Retrieval-Augmented Generation) et embeddings",
    version="1.0.0"
)

# CORS : Cross-Origin Resource Sharing
# Nécessaire pour que ton frontend (ou Flutter) puisse appeler l'API
# depuis un domaine différent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossier où stocker les fichiers uploadés
DOSSIER_UPLOAD = Path("data")
DOSSIER_UPLOAD.mkdir(exist_ok=True) # Créer le dossier s'il n'existe pas

# Modèles Pydantic pour la validation des données

class QuestionRequest(BaseModel):
    """Modèle pour une question de l'utilisateur"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="La question à poser au chatbot"
    )
    historique: list[dict] = Field(
        default=[],
        description="Historique optionnel des échanges précédents"
    )

class ReponseChatbot(BaseModel):
    """Modèle pour la réponse du chatbot"""
    reponse: str
    sources: list[str]
    tokens_utilises: int

# ROUTE 1 : Vérification que l'API fonctionne
@app.get("/", tags=["Santé"])
async def racine():
    """Endpoint de vérification. Utile pour les healthchecks."""
    return {
        "statut": "opérationnel",
        "message": "API Chatbot FAQ prête à recevoir des questions."
    }

# ROUTE 2 : Upload et indexation d'un document
@app.post("/document/upload", tags=["Documents"])
async def uploader_document(fichier: UploadFile = File(...)):
    """
    Upload un fichier (.txt ou .pdf) et l'indexe dans ChromaDB.
    C'est la phase d'initialisation du RAG.

    :param fichier: Le fichier uploadé via multipart/form-data

    :return:
    """
    # Vérifier l'extension du fichier
    extension_autorisees = [".txt", ".pdf", ".md"]
    extension = Path(fichier.filename).suffix.lower()

    if extension not in extension_autorisees:
        raise HTTPException (
            status_code=400,
            detail=f"Format non supporté. Formats acceptés : {', '.join(extension_autorisees)}"
        )
    # Sauvegarder le fichier uploadé sur le disque
    chemin_fichier = DOSSIER_UPLOAD / fichier.filename
    with open(chemin_fichier, "wb") as buffer:
        #shutil.copyfileobj copie les données du fichier uploadé vers le buffer
        shutil.copyfileobj(fichier.file, buffer)

    # Charger et découper le document en chunks
    chunks = charger_document(str(chemin_fichier))

    # Indexer les chunks dans ChromaDB
    indexer_documents(chunks, nom_source=fichier.filename)

    return {
        "message": f"Document '{fichier.filename}' indexé avec succès",
        "nombre_chunks": len(chunks)
    }

# ROUTE 3 : Poser une question au chatbot
@app.post("/chat", response_model=ReponseChatbot, tags=["Chat"])
async def chat(request: QuestionRequest):
    """
    Endpoint principal du chatbot.
    Reçoit une question, cherche dans ChromaDB, génère une réponse avec GPT.

    Args:
        request: Question + historique optionnel

    Returns:
        Réponse du chatbot + sources utilisées
    """
    # Validation supplémentaire : vérifier que la base n'est pas vide
    # (l'utilisateur doit d'abord uploader un document)
    try:
        resultat = repondre_a_question(
            question=request.question,
            historique=request.historique
        )
        return ReponseChatbot(**resultat)

    except Exception as e:
        # En cas d'erreur (API OpenAI down, ChromaDB vide, etc.)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement : {str(e)}"
       )