# Ce module gère toute l'interaction avec ChromaDB :
# - Initialisation de la base
# - Ajout de documents
# - Recherche par similarité
from importlib.metadata import metadata
from xml.etree.ElementInclude import include

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE D'EMBEDDING (une seule fois au démarrage)
# ─────────────────────────────────────────────────────────────────────────────
# SentenceTransformer charge le modèle depuis le cache local (ou le télécharge
# depuis Hugging Face si c'est la première fois, ~570MB).
# On charge UNE SEULE FOIS au démarrage du module, pas à chaque requête.
# Si on le rechargeait à chaque appel, chaque question prendrait 5-10s de plus.

NOM_MODELE_EMBEDDING = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

print(f"⌛Chargement du modèle d'embedding : {NOM_MODELE_EMBEDDING} ...")
modele_embedding = SentenceTransformer(NOM_MODELE_EMBEDDING)
print(f"Modèle BGE-M3 prêt !")


def creer_client_chroma(chemin_db: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    Crée ou ouvre une base ChromaDB persistante sur le disque.

    PersistentClient = les données sont sauvegardées entre les redémarrages.
    (vs EphemeralClient qui stocke tout en mémoire,perdu au redémarrage)
    :param chemin_db: Dossier où stocker la base (créé automatiquement)
    :return: Client ChromaDB configuré
    """

    return chromadb.PersistentClient(path=chemin_db)

def obtenir_collection(client: chromadb.PersistentClient, nom: str ="faq"):
    """
    Récupère une collection existante ou la crée si elle n'existe pas.
    get_or_create_collection = idempotent (safe à appeler plusieurs fois)
    :param client: Le client ChromaDB
    :param nom: Nom de la collection (comme un nom de table en SQL)
    :return: La collection ChromaDB
    """

    return client.get_or_create_collection(
        name = nom,
        metadata={"hnsw:space": "cosine"}
        # "cosine" = mesure de similarité cosinus
        # Obligatoire quand on utilise normalize_embeddings=True

    )

def generer_embedding(texte: str) -> list[float]:
    """
    Convertit un texte en vecteur numérique (embedding) avec BGE-M3 LOCAL.
    Aucun appel réseau, aucun coût, tourne directement sur ton CPU/GPU.

    BGE-M3 produit des vecteurs de 1024 dimensions.
    (OpenAI text-embedding-3-small en produisait 1536 — ChromaDB s'adapte)

    :param texte: Le texte à convertir en vecteur

    :return: Liste de 1024 nombres représentant le texte
    """

    vecteur = modele_embedding.encode(
        texte,
        normalize_embeddings=True
        # normalize_embeddings=True :
        # Force la norme du vecteur à 1 (vecteur unitaire).
        # Indispensable pour la similarité cosinus soit fiable.
        # Sans normalisation, des textes longs auraient des vecteurs de grandes
        # normes, faussant la comparaison
    )
    # encode() retourne un numpy.ndarray -> on le convertit en list Python
    # car ChromaDB attend une list[float]
    return vecteur.tolist()

def generer_emdedding_batch(textes: list[str]) -> list[list[float]]:
    """
    Génère les embeddings pour une liste de textes en une seule passe.
    BEAUCOUP plus efficace que d'appeler generer_embedding() pour chaque texte en boucle.
    BGE-M3 peut traiter des dizaines de textes simultanément sur GPU.
    :param textes: Liste de textes à encoder
    :return:
        Liste de vecteurs (un par texte)
    """

    vecteurs = modele_embedding.encode(
        textes, # Liste entière d'un coup
        normalize_embeddings=True,
        batch_size=32, # Traiter 32 textes à la fois
        show_progress_bar=True # Afficher un barre de progression
    )
    return [v.tolist() for v in vecteurs]

def  indexer_documents(chunks: list[str], nom_source: str = "document"):
    """
    Prend des chunks de texte, génère leurs embeddings et les stocke dans ChromaDB
    C'est la PHASE 1 du RAG : L'indexation.

    Optimisation : on utilise generer_embeddings_batch() pour générer les embeddings
    en seule passe GPU
    :param chunks: Liste des morceaux de texte à indexer
    :param nom_source: Nom du document source (pour les métadonnées)
    :return:
    """

    client_chroma = creer_client_chroma()
    collection = obtenir_collection(client_chroma)

    # Vérifier si la collection est déjà remplie
    if collection.count() > 0:
        print(f" Collection déjà remplie ({collection.count()} chunks) !")
        # On supprime et recrée la collection pour repartir proprement
        client_chroma.delete_collection("faq")
        collection = obtenir_collection(client_chroma)

    print(f" Génération des embeddings pour {len(chunks)} chunks ...")

    # OPTIMISATION : Générer TOUS les embeddings en une seule passe batch
    # Au lieu de : for chunk in chunks: generer_embedding(chunk) <- lent
    # On fait : generer_embeddings_batch(chunks)
    tous_les_vecteurs = generer_emdedding_batch(chunks)

    # Préparer les métadonnées et identifiants
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
    metadatas = [{"source": nom_source, "index": i} for i in range(len(chunks))]

    # Insérer tous les chunks en une seule opération atomique
    collection.add(
        documents=chunks,
        embeddings=tous_les_vecteurs,
        ids=ids,
        metadatas=metadatas
    )

    print(f"{len(chunks)} chunks indexés dans ChromaDB !")

def rechercher_documents_pertinents(question: str, n_resultats: int = 3) -> list[str]:
    """
    Cherche les chunks les plus pertinents pour une question donnée.
    C'est la première partie de la PHASE 2 du RAG : le Retrieval.

    :param question: La question de l'utilisateur
    :param n_resultats: Nombre du chunks à retourner (3 est un bon défaut)

    :return: Liste des chunks les plus proches de la question
    """

    client_chroma = creer_client_chroma()
    collection = obtenir_collection(client_chroma)

    # Convertir la question en vecteur avec BGE-M3 (même modèle que l'indexation)
    # IMPORTANT : toujours utiliser le même modèle pour indexer et pour chercher.
    # Mélanger les modèles donnerait des résultats incohérents.
    vecteur_question = generer_embedding(question)

    # Chercher les n_resultats les plus proches dans la ChromaDB
    resultats = collection.query(
        query_embeddings=[vecteur_question], # Liste de vecteurs
        n_results=min(n_resultats, collection.count()), # Sécurité : ne pas demander plus que ce qui existe
        include=["documents","distances","metadatas"]
    )

    # resultats["documents"] est une liste de listes (une liste par vecteur de requête)
    # Comme on a envoyé 1 vecteur : resultats["documents"][0] = nos chunks
    chunks_trouves = resultats["documents"][0]

    # Pour le débogage, tu peux afficher les scores de similarité :
    # distances = resultats["distances"][0]
    # for chunk, dist in zip(chunks_trouves, distances):
    #   print(f"Score: {1-dist:.3f} | {chunk[:80]}...")

    return chunks_trouves