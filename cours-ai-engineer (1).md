# Cours Complet — AI Engineer Débutant
## Python · FastAPI · ChromaDB · Ollama · BAAI/bge-m3 · Mistral · LangChain · PostgreSQL · pgvector · LangGraph · Flutter

> **Comment utiliser ce cours**
> Chaque projet est autonome. Lis d'abord la section "Fondations théoriques", ensuite le guide de réalisation pas à pas, puis le code annoté. Les 3 niveaux d'explication sont présents : ligne par ligne, par blocs logiques, et concepts globaux.

---

# TABLE DES MATIÈRES

- [PROJET 1 — Chatbot FAQ avec FastAPI, ChromaDB, Ollama et BGE-M3](#projet-1)
- [PROJET 2 — Assistant Text-to-SQL avec LangChain et SQLite](#projet-2)
- [PROJET 3 — App Flutter de chatbot IA avec historique](#projet-3)
- [PROJET 4 — Agent d'analyse de documents avec pgvector](#projet-4)
- [PROJET 5 — Système multi-agents avec LangGraph](#projet-5)

---

# PROJET 1 — Chatbot FAQ avec FastAPI, ChromaDB, Ollama et BGE-M3 {#projet-1}

## 1.1 Comprendre le problème qu'on résout

Une entreprise (école, pharmacie, boutique) reçoit les mêmes questions tous les jours : "Quels sont vos horaires ?", "Comment s'inscrire ?", "Quel est le prix de X ?". Au lieu de répondre manuellement, on construit un système qui :

1. Lit un document existant (le règlement intérieur, la liste de prix, etc.)
2. Comprend la question de l'utilisateur
3. Trouve la bonne partie du document
4. Génère une réponse naturelle en français

Ce système s'appelle **RAG** : Retrieval-Augmented Generation.

---

## 1.2 Notion fondamentale : Le RAG (Retrieval-Augmented Generation)

### Qu'est-ce que c'est ?

Un LLM (comme GPT-4) a été entraîné sur des milliards de textes, mais il ne connaît PAS le règlement intérieur de l'Université de Yaoundé ou la liste des médicaments de ta pharmacie.

Le RAG résout ce problème en deux temps :
1. **Retrieval** : Récupérer les informations pertinentes depuis une base de données
2. **Generation** : Utiliser ces informations comme contexte pour générer une réponse

### Analogie

Imagine un étudiant qui passe un examen avec le droit d'avoir ses notes. Avant de répondre, il cherche dans ses notes la partie pertinente, puis formule sa réponse. Le RAG, c'est exactement ça.

### Architecture du RAG

```
PHASE 1 — INDEXATION (une seule fois au départ)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Document    │───▶│   Découpage  │───▶│  Embeddings  │───▶│  ChromaDB    │
│  (PDF/TXT)   │    │  en chunks   │    │  (vecteurs)  │    │  (stockage)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘

PHASE 2 — INTERROGATION (à chaque question)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Question    │───▶│  Embedding   │───▶│  Recherche   │───▶│  Mistral 7B  │───▶ Réponse
│  utilisateur │    │  BGE-M3      │    │  similarité  │    │  (Ollama)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 1.3 Notion fondamentale : Les Embeddings

### Définition

Un embedding est une représentation numérique d'un texte sous forme de vecteur (une liste de nombres). Des textes avec des sens similaires ont des vecteurs proches dans l'espace mathématique.

### Exemple concret

```
"Quels sont vos horaires d'ouverture ?"
→ [0.23, -0.45, 0.12, 0.89, ..., 0.03]  (1024 nombres avec BGE-M3)

"À quelle heure êtes-vous ouverts ?"
→ [0.21, -0.44, 0.14, 0.87, ..., 0.05]  (très proche du précédent !)

"Quel est le prix du paracétamol ?"
→ [-0.67, 0.34, -0.23, 0.12, ..., 0.78]  (très différent)
```

Deux questions qui veulent dire la même chose auront des vecteurs proches. C'est ce qui permet de trouver la bonne réponse même si l'utilisateur ne pose pas la question exactement comme dans le document.

### Similarité cosinus

Pour comparer deux vecteurs, on utilise la **similarité cosinus** : elle mesure l'angle entre deux vecteurs. Un angle de 0° (vecteurs identiques) = similarité de 1. Un angle de 90° = similarité de 0.

```python
# Voilà ce qui se passe mathématiquement derrière ChromaDB
import numpy as np

def similarite_cosinus(vecteur_a, vecteur_b):
    # Produit scalaire divisé par le produit des normes
    produit_scalaire = np.dot(vecteur_a, vecteur_b)
    norme_a = np.linalg.norm(vecteur_a)
    norme_b = np.linalg.norm(vecteur_b)
    return produit_scalaire / (norme_a * norme_b)

# Plus le résultat est proche de 1, plus les textes sont similaires
```

---

## 1.4 Notion fondamentale : FastAPI

### Qu'est-ce qu'une API ?

Une API (Application Programming Interface) est un contrat entre deux programmes. Elle définit comment un programme peut demander quelque chose à un autre.

Quand tu poses une question dans le chat, ton interface envoie une **requête HTTP** au serveur, qui répond avec la réponse du bot.

### Pourquoi FastAPI ?

FastAPI est un framework Python moderne qui permet de créer des APIs rapidement. Ses avantages :
- **Rapide** : Basé sur Starlette et Pydantic, c'est l'un des frameworks Python les plus performants
- **Documentation automatique** : Il génère une interface Swagger à `/docs` automatiquement
- **Validation des données** : Grâce à Pydantic, les données sont vérifiées avant d'entrer dans ton code
- **Async natif** : Gère plusieurs requêtes en même temps

### Les concepts clés de FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Créer l'application
app = FastAPI()

# 2. Définir un modèle de données (validation automatique)
class QuestionRequest(BaseModel):
    question: str       # Obligatoire, doit être une chaîne
    langue: str = "fr"  # Optionnel, valeur par défaut "fr"

# 3. Créer une route (endpoint)
# @app.post signifie : "quand quelqu'un envoie une requête POST à /chat"
@app.post("/chat")
async def chat(request: QuestionRequest):
    # request.question est déjà validé par Pydantic
    # Si ce n'est pas une string, FastAPI retourne automatiquement une erreur 422
    return {"reponse": f"Tu as demandé : {request.question}"}

# 4. Route GET simple
@app.get("/")
async def racine():
    return {"message": "API Chatbot FAQ opérationnelle"}
```

### Les méthodes HTTP

| Méthode | Utilisation | Exemple |
|---------|-------------|---------|
| GET | Récupérer des données | `/documents` → liste les docs |
| POST | Envoyer des données | `/chat` → envoyer une question |
| PUT | Modifier des données | `/document/1` → mettre à jour |
| DELETE | Supprimer | `/document/1` → supprimer |

---

## 1.5 Notion fondamentale : ChromaDB

### Qu'est-ce qu'une base vectorielle ?

Une base de données classique stocke des valeurs exactes et cherche des correspondances exactes :
```sql
SELECT * FROM faq WHERE question = "Quels sont vos horaires ?"
```
Problème : si l'utilisateur écrit "C'est ouvert à quelle heure ?", ça ne trouve rien.

Une base **vectorielle** comme ChromaDB stocke des embeddings (vecteurs) et cherche par **similarité** :
```python
# Trouve les 3 chunks les plus proches de la question
resultats = collection.query(
    query_texts=["C'est ouvert à quelle heure ?"],
    n_results=3
)
# Retourne les chunks parlant des horaires, même avec des mots différents
```

### Les concepts de ChromaDB

```python
import chromadb

# 1. Créer un client (base locale stockée dans un dossier)
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Créer une collection (comme une table en SQL)
collection = client.get_or_create_collection(
    name="faq_documents",
    # La fonction de distance utilisée pour comparer les vecteurs
    metadata={"hnsw:space": "cosine"}
)

# 3. Ajouter des documents
collection.add(
    documents=["Nous sommes ouverts de 8h à 18h du lundi au vendredi"],
    ids=["chunk_001"],  # Identifiant unique obligatoire
    metadatas=[{"source": "reglement.pdf", "page": 1}]  # Infos optionnelles
)

# 4. Interroger
resultats = collection.query(
    query_texts=["horaires d'ouverture"],
    n_results=2  # Les 2 chunks les plus pertinents
)
print(resultats["documents"])    # Le texte des chunks
print(resultats["distances"])    # Score de similarité
print(resultats["metadatas"])    # Les métadonnées associées
```

---

## 1.6 Notion fondamentale : Ollama + BGE-M3 (stack 100% gratuit)

Au lieu d'utiliser l'API payante d'OpenAI, on utilise deux modèles open source qui tournent **entièrement sur ton PC**, sans connexion internet, sans coût.

### Pourquoi cette stack ?

| | OpenAI (payant) | Notre stack (gratuit) |
|---|---|---|
| **Embeddings** | text-embedding-3-small | `BAAI/bge-m3` (Hugging Face) |
| **Génération** | gpt-4o-mini | `Mistral 7B` via Ollama |
| **Coût** | ~$0.002/conversation | $0.00 |
| **Internet** | Requis | Non requis |
| **Vitesse (RTX 4060)** | Dépend du réseau | 40-50 tokens/seconde |

---

### Notion : BAAI/bge-m3 (modèle d'embedding)

BGE-M3 est le modèle d'embedding le plus populaire de Hugging Face. Ses avantages clés :

- **Multilingue** : Supporte 100+ langues dont le français et les langues locales
- **Léger** : ~570MB, tourne sur CPU, n'occupe presque pas la VRAM
- **Gratuit** : Licence Apache 2.0, utilisation commerciale autorisée
- **Dimensions** : Produit des vecteurs de **1024 dimensions** (vs 1536 pour OpenAI)

```python
from sentence_transformers import SentenceTransformer

# Charger le modèle (téléchargement automatique au 1er lancement ~570MB)
modele = SentenceTransformer("BAAI/bge-m3")

# Générer un embedding
texte = "Quels sont vos horaires d'ouverture ?"
vecteur = modele.encode(texte, normalize_embeddings=True)
# vecteur = array de 1024 floats

print(f"Dimensions : {len(vecteur)}")  # 1024
print(f"Type : {type(vecteur)}")       # numpy.ndarray
```

**Pourquoi `normalize_embeddings=True` ?**
La normalisation force la norme du vecteur à 1. Ça rend la similarité cosinus plus stable et les comparaisons plus précises.

---

### Notion : Ollama (moteur d'inférence local)

Ollama est un outil qui permet de faire tourner des LLMs localement en quelques commandes. Il gère automatiquement :
- Le téléchargement et stockage des modèles
- La **quantification** (compression du modèle pour tenir en VRAM)
- L'exposition d'une **API compatible OpenAI** sur `localhost:11434`

**Qu'est-ce que la quantification ?**
Un modèle Mistral 7B en précision complète (float32) = ~28GB. Impossible sur 8GB de VRAM. La quantification en 4 bits (Q4_K_M) le compresse à ~4.1GB avec environ 95% des capacités conservées.

```
Mistral 7B original (float32) : 28 GB  ❌ Ne rentre pas
Mistral 7B Q8                 :  7 GB  ✅ Juste
Mistral 7B Q4_K_M             :  4.1 GB ✅ Confortable, recommandé
```

**Installation et utilisation :**

```bash
# 1. Télécharger Ollama depuis https://ollama.com (Windows/Linux/Mac)

# 2. Télécharger Mistral (Q4_K_M par défaut, ~4.1GB)
ollama pull mistral

# 3. Lancer le serveur Ollama
ollama serve
# → Écoute sur http://localhost:11434

# 4. Tester en ligne de commande
ollama run mistral "Réponds en français : quels sont tes capacités ?"

# 5. Voir les modèles installés
ollama list
```

**La magie d'Ollama : compatibilité OpenAI**
Ollama expose une API identique à celle d'OpenAI. Tu utilises le même SDK Python, tu changes juste l'URL et le nom du modèle :

```python
from openai import OpenAI

# Client pointant vers Ollama LOCAL au lieu d'OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",  # URL Ollama
    api_key="ollama"                        # Valeur factice, ignorée par Ollama
)

# Appel identique à OpenAI
reponse = client.chat.completions.create(
    model="mistral",   # Nom du modèle Ollama (au lieu de "gpt-4o-mini")
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant FAQ. Réponds uniquement en français."
        },
        {
            "role": "user",
            "content": "Quels sont les horaires ?"
        }
    ],
    temperature=0.1,
)

texte = reponse.choices[0].message.content
print(texte)
```

### Qu'est-ce qu'un token ?

Un token est une unité de texte. En gros : 1 mot ≈ 1.3 tokens. "Bonjour comment allez-vous" = environ 5 tokens. Les LLMs lisent et génèrent des tokens, pas des mots.

### Les rôles dans les messages

- `system` : Définit le comportement global du modèle. Invisible à l'utilisateur.
- `user` : Ce que l'utilisateur dit.
- `assistant` : Ce que le modèle a répondu (pour l'historique de conversation).

### Modèles Ollama alternatifs pour ta RTX 4060

| Modèle | Taille VRAM | Vitesse | Qualité français | Usage |
|--------|-------------|---------|-----------------|-------|
| `mistral` | 4.1 GB | 40-50 tok/s | ⭐⭐⭐⭐ | Recommandé |
| `mistral-small3` | 4.5 GB | 35-45 tok/s | ⭐⭐⭐⭐⭐ | Meilleure qualité |
| `llama3.2` | 2.0 GB | 55-65 tok/s | ⭐⭐⭐ | Si tu veux de la vitesse |
| `phi4-mini` | 2.5 GB | 50-60 tok/s | ⭐⭐⭐ | Compact et efficace |

---

## 1.7 Structure complète du projet

```
chatbot-faq/
│
├── app/
│   ├── __init__.py          # Indique que ce dossier est un package Python
│   ├── main.py              # Point d'entrée FastAPI
│   ├── ingestion.py         # Chargement et découpage des documents
│   ├── embeddings.py        # Gestion de ChromaDB + BGE-M3 (local)
│   └── chat.py              # Logique de conversation avec Mistral via Ollama
│
├── data/
│   └── faq.txt              # Ton document source (ou PDF)
│
├── chroma_db/               # Créé automatiquement par ChromaDB
│
├── .env                     # Variables d'environnement (URL Ollama)
├── requirements.txt         # Dépendances Python
└── README.md
```

> ⚠️ **Prérequis** : Ollama doit être installé et en cours d'exécution (`ollama serve`) avant de lancer le projet. Le modèle BGE-M3 se télécharge automatiquement depuis Hugging Face au premier lancement.

---

## 1.8 Code complet annoté

### `requirements.txt`

```
fastapi==0.111.0               # Framework web
uvicorn==0.29.0                # Serveur ASGI pour lancer FastAPI
openai==1.30.0                 # SDK OpenAI (réutilisé pour appeler Ollama !)
chromadb==0.5.0                # Base de données vectorielle
sentence-transformers==3.0.0   # Pour charger BGE-M3 depuis Hugging Face
pdfplumber==0.11.0             # Extraction de texte depuis PDF
python-dotenv==1.0.1           # Chargement des variables d'environnement
python-multipart==0.0.9        # Gestion de l'upload de fichiers
torch==2.3.0                   # Requis par sentence-transformers (GPU)
```

> **Note sur torch** : Si tu as une carte NVIDIA, installe la version CUDA de PyTorch pour accélérer la génération des embeddings :
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### `.env`

```bash
# Plus besoin de clé API OpenAI !
# On pointe juste vers le serveur Ollama local

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=BAAI/bge-m3

# Si tu passes à un autre modèle Ollama, change juste OLLAMA_MODEL
# Ex : OLLAMA_MODEL=mistral-small3
```

### `data/faq.txt` (exemple)

```
ÉCOLE DE FORMATION INFORMATIQUE DE YAOUNDÉ
FAQ OFFICIELLE

HORAIRES
L'école est ouverte du lundi au vendredi de 8h00 à 18h00.
Le samedi de 9h00 à 13h00.
L'école est fermée les dimanches et jours fériés.

INSCRIPTIONS
Les inscriptions se font du 1er au 30 septembre de chaque année.
Les frais d'inscription sont de 50 000 FCFA.
Les documents requis sont : photocopie du baccalauréat, 2 photos d'identité, extrait de naissance.

FORMATIONS DISPONIBLES
Développement web : durée 12 mois, coût 450 000 FCFA/an
Intelligence artificielle : durée 18 mois, coût 600 000 FCFA/an
Administration réseau : durée 12 mois, coût 400 000 FCFA/an

CONTACT
Téléphone : +237 6XX XXX XXX
Email : info@ecole-info-yaounde.cm
Adresse : Quartier Bastos, Rue 1234, Yaoundé
```

### `app/ingestion.py` — Chargement et découpage

```python
# Ce module s'occupe de lire les documents et de les découper en morceaux (chunks)

import pdfplumber          # Pour lire les PDFs
from pathlib import Path   # Pour manipuler les chemins de fichiers

def lire_fichier_texte(chemin: str) -> str:
    """
    Lit un fichier .txt et retourne son contenu.

    Args:
        chemin: Le chemin vers le fichier, ex: "data/faq.txt"

    Returns:
        Le contenu du fichier sous forme de chaîne de caractères
    """
    # Path() permet de manipuler les chemins de façon portable (Windows/Linux/Mac)
    chemin_fichier = Path(chemin)

    # Vérification que le fichier existe avant de le lire
    if not chemin_fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    # encoding="utf-8" est important pour les caractères spéciaux (accents, etc.)
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return f.read()


def lire_pdf(chemin: str) -> str:
    """
    Extrait tout le texte d'un fichier PDF.

    Args:
        chemin: Le chemin vers le fichier PDF

    Returns:
        Tout le texte du PDF concatené
    """
    texte_complet = []

    # pdfplumber ouvre le PDF et permet d'accéder page par page
    with pdfplumber.open(chemin) as pdf:
        for numero_page, page in enumerate(pdf.pages):
            # extract_text() retourne le texte de la page ou None si la page est une image
            texte = page.extract_text()
            if texte:  # On ignore les pages sans texte (images, pages vides)
                texte_complet.append(texte)

    # "\n\n" entre les pages pour marquer les séparations
    return "\n\n".join(texte_complet)


def decouper_en_chunks(texte: str, taille: int = 500, chevauchement: int = 50) -> list[str]:
    """
    Découpe un long texte en morceaux de taille fixe avec chevauchement.

    Pourquoi le chevauchement ?
    Si on découpe sans chevauchement, une information importante peut se retrouver
    coupée entre deux chunks. Le chevauchement assure la continuité du contexte.

    Exemple avec taille=20, chevauchement=5 :
    Texte : "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    Chunk 1 : "ABCDEFGHIJKLMNOPQRST"  (positions 0-19)
    Chunk 2 : "PQRSTUVWXYZ..."        (positions 15-34, 5 caractères en commun)

    Args:
        texte: Le texte source à découper
        taille: Nombre de caractères par chunk (500 est un bon compromis)
        chevauchement: Nombre de caractères partagés entre chunks consécutifs

    Returns:
        Liste de chunks (morceaux de texte)
    """
    chunks = []

    # On commence au début du texte
    debut = 0

    while debut < len(texte):
        # La fin du chunk est soit début+taille, soit la fin du texte
        fin = min(debut + taille, len(texte))

        # Extraire le morceau de texte
        chunk = texte[debut:fin].strip()  # strip() enlève les espaces en début/fin

        # On ignore les chunks trop courts (moins de 20 caractères)
        # Ils ne contiennent généralement pas assez d'information
        if len(chunk) >= 20:
            chunks.append(chunk)

        # Le prochain chunk commence AVANT la fin du chunk actuel (chevauchement)
        debut += taille - chevauchement

    return chunks


def charger_document(chemin: str) -> list[str]:
    """
    Fonction principale : charge un document et retourne la liste de chunks.
    Détecte automatiquement si c'est un PDF ou un fichier texte.

    Args:
        chemin: Chemin vers le fichier (.txt ou .pdf)

    Returns:
        Liste de chunks prêts à être indexés
    """
    # Détecter le type de fichier via son extension
    extension = Path(chemin).suffix.lower()  # ".pdf" ou ".txt"

    if extension == ".pdf":
        texte = lire_pdf(chemin)
    elif extension in [".txt", ".md"]:
        texte = lire_fichier_texte(chemin)
    else:
        raise ValueError(f"Format non supporté : {extension}. Utilise .pdf ou .txt")

    # Découper le texte en chunks
    chunks = decouper_en_chunks(texte)

    print(f"✅ Document chargé : {len(chunks)} chunks créés depuis '{chemin}'")
    return chunks
```

### `app/embeddings.py` — Gestion de ChromaDB + BGE-M3

```python
# Ce module gère toute l'interaction avec ChromaDB :
# - Initialisation de la base vectorielle
# - Génération des embeddings avec BGE-M3 (LOCAL, gratuit)
# - Ajout de documents
# - Recherche par similarité

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
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

print(f"⏳ Chargement du modèle d'embedding '{NOM_MODELE_EMBEDDING}'...")
modele_embedding = SentenceTransformer(NOM_MODELE_EMBEDDING)
print(f"✅ Modèle BGE-M3 prêt !")
# ─────────────────────────────────────────────────────────────────────────────


def creer_client_chroma(chemin_db: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    Crée ou ouvre une base ChromaDB persistante sur le disque.

    PersistentClient = les données sont sauvegardées entre les redémarrages.
    (vs EphemeralClient qui stocke tout en mémoire, perdu au redémarrage)

    Args:
        chemin_db: Dossier où stocker la base (créé automatiquement)

    Returns:
        Client ChromaDB configuré
    """
    return chromadb.PersistentClient(path=chemin_db)


def obtenir_collection(client: chromadb.PersistentClient, nom: str = "faq"):
    """
    Récupère une collection existante ou la crée si elle n'existe pas.
    get_or_create_collection = idempotent (safe à appeler plusieurs fois)

    Args:
        client: Le client ChromaDB
        nom: Nom de la collection (comme un nom de table en SQL)

    Returns:
        La collection ChromaDB
    """
    return client.get_or_create_collection(
        name=nom,
        metadata={"hnsw:space": "cosine"}
        # "cosine" = mesure de similarité cosinus
        # Obligatoire quand on utilise normalize_embeddings=True dans BGE-M3
    )


def generer_embedding(texte: str) -> list[float]:
    """
    Convertit un texte en vecteur numérique (embedding) avec BGE-M3 LOCAL.
    Aucun appel réseau, aucun coût, tourne directement sur ton CPU/GPU.

    BGE-M3 produit des vecteurs de 1024 dimensions.
    (OpenAI text-embedding-3-small en produisait 1536 — ChromaDB s'adapte)

    Args:
        texte: Le texte à convertir en vecteur

    Returns:
        Liste de 1024 floats représentant le texte
    """
    vecteur = modele_embedding.encode(
        texte,
        normalize_embeddings=True
        # normalize_embeddings=True :
        # Force la norme du vecteur à 1 (vecteur unitaire).
        # Indispensable pour que la similarité cosinus soit fiable.
        # Sans normalisation, des textes longs auraient des vecteurs de grande
        # norme, faussant les comparaisons.
    )
    # encode() retourne un numpy.ndarray → on le convertit en list Python
    # car ChromaDB attend une list[float]
    return vecteur.tolist()


def generer_embeddings_batch(textes: list[str]) -> list[list[float]]:
    """
    Génère les embeddings pour une liste de textes en une seule passe.
    BEAUCOUP plus efficace que d'appeler generer_embedding() en boucle.
    BGE-M3 peut traiter des dizaines de textes simultanément sur GPU.

    Args:
        textes: Liste de textes à encoder

    Returns:
        Liste de vecteurs (un par texte)
    """
    vecteurs = modele_embedding.encode(
        textes,                    # Liste entière d'un coup
        normalize_embeddings=True,
        batch_size=32,             # Traiter 32 textes à la fois (optimal pour 8GB VRAM)
        show_progress_bar=True     # Affiche une barre de progression dans le terminal
    )
    return [v.tolist() for v in vecteurs]


def indexer_documents(chunks: list[str], nom_source: str = "document"):
    """
    Prend des chunks de texte, génère leurs embeddings et les stocke dans ChromaDB.
    C'est la PHASE 1 du RAG : l'indexation.

    Optimisation : on utilise generer_embeddings_batch() pour générer tous
    les embeddings en une seule passe GPU (bien plus rapide que chunk par chunk).

    Args:
        chunks: Liste des morceaux de texte à indexer
        nom_source: Nom du document source (pour les métadonnées)
    """
    client_chroma = creer_client_chroma()
    collection = obtenir_collection(client_chroma)

    # Vérifier si la collection est déjà remplie
    if collection.count() > 0:
        print(f"⚠️  Collection déjà remplie ({collection.count()} chunks). Réinitialisation...")
        # On supprime et recrée la collection pour repartir proprement
        client_chroma.delete_collection("faq")
        collection = obtenir_collection(client_chroma)

    print(f"🔄 Génération des embeddings pour {len(chunks)} chunks avec BGE-M3...")

    # OPTIMISATION : Générer TOUS les embeddings en une seule passe batch
    # Au lieu de : for chunk in chunks: generer_embedding(chunk)  ← lent
    # On fait   : generer_embeddings_batch(chunks)               ← rapide
    tous_les_vecteurs = generer_embeddings_batch(chunks)

    # Préparer les métadonnées et identifiants
    ids       = [f"chunk_{i:04d}" for i in range(len(chunks))]
    metadatas = [{"source": nom_source, "index": i} for i in range(len(chunks))]

    # Insérer tous les chunks en une seule opération atomique
    collection.add(
        documents=chunks,
        embeddings=tous_les_vecteurs,
        ids=ids,
        metadatas=metadatas
    )

    print(f"✅ {len(chunks)} chunks indexés dans ChromaDB !")


def rechercher_chunks_pertinents(question: str, n_resultats: int = 3) -> list[str]:
    """
    Cherche les chunks les plus pertinents pour une question donnée.
    C'est la première partie de la PHASE 2 du RAG : le Retrieval.

    Args:
        question: La question de l'utilisateur
        n_resultats: Nombre de chunks à retourner (3 est un bon défaut)

    Returns:
        Liste des chunks les plus proches de la question
    """
    client_chroma = creer_client_chroma()
    collection = obtenir_collection(client_chroma)

    # Convertir la question en vecteur avec BGE-M3 (même modèle que l'indexation !)
    # IMPORTANT : toujours utiliser le même modèle pour indexer et pour chercher.
    # Mélanger les modèles donnerait des résultats incohérents.
    vecteur_question = generer_embedding(question)

    # Chercher les n_resultats chunks les plus proches dans ChromaDB
    resultats = collection.query(
        query_embeddings=[vecteur_question],  # Liste de vecteurs (ici on en envoie un seul)
        n_results=min(n_resultats, collection.count()),  # Sécurité : ne pas demander plus que ce qui existe
        include=["documents", "distances", "metadatas"]
    )

    # resultats["documents"] est une liste de listes (une liste par vecteur de requête)
    # Comme on a envoyé 1 vecteur : resultats["documents"][0] = nos chunks
    chunks_trouves = resultats["documents"][0]

    # Pour le débogage, tu peux afficher les scores de similarité :
    # distances = resultats["distances"][0]
    # for chunk, dist in zip(chunks_trouves, distances):
    #     print(f"Score: {1-dist:.3f} | {chunk[:80]}...")

    return chunks_trouves
```

### `app/chat.py` — Logique de conversation avec Mistral via Ollama

```python
# Ce module orchestre la conversation :
# 1. Prend la question de l'utilisateur
# 2. Récupère les chunks pertinents depuis ChromaDB
# 3. Construit le prompt avec le contexte
# 4. Appelle Mistral 7B via Ollama (LOCAL, gratuit)
# 5. Retourne la réponse

from openai import OpenAI   # On réutilise le SDK OpenAI, mais pointé vers Ollama !
from dotenv import load_dotenv
import os
from .embeddings import rechercher_chunks_pertinents

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
    api_key="ollama"  # Ollama ignore cette valeur, mais le SDK l'exige
)

NOM_MODELE = os.getenv("OLLAMA_MODEL", "mistral")
# ─────────────────────────────────────────────────────────────────────────────


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
    Fonction principale : génère une réponse à une question en utilisant le RAG.

    Flux complet :
    1. Récupérer les chunks pertinents depuis ChromaDB (Retrieval)
    2. Construire le prompt avec ces chunks comme contexte
    3. Appeler Mistral 7B via Ollama (Generation)
    4. Retourner la réponse + les sources utilisées

    Args:
        question: La question de l'utilisateur
        historique: Liste optionnelle des échanges précédents (pour la mémoire)
                    Format : [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Returns:
        Dictionnaire avec "reponse" (str), "sources" (list[str]) et "modele" (str)
    """
    if historique is None:
        historique = []

    # ÉTAPE 1 : Récupérer les chunks pertinents (Retrieval avec BGE-M3 + ChromaDB)
    chunks_pertinents = rechercher_chunks_pertinents(question, n_resultats=3)

    # ÉTAPE 2 : Construire le contexte depuis les chunks
    # On joint les chunks avec des séparateurs clairs pour aider Mistral
    # à distinguer les différentes sections du contexte
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
        model=NOM_MODELE,      # "mistral" (ou ce qui est dans .env)
        messages=messages,
        temperature=0.1,       # Très bas = réponses factuelles et cohérentes
                               # 0 = toujours la même réponse (déterministe)
                               # 1 = créatif et varié (pas adapté pour FAQ)
        # Note : pas de max_tokens ici car Mistral gère bien la longueur
        # avec un bon prompt. Tu peux l'ajouter si les réponses sont trop longues.
    )

    # Extraire le texte de la réponse (même structure qu'OpenAI)
    texte_reponse = reponse_api.choices[0].message.content

    return {
        "reponse": texte_reponse,
        "sources": chunks_pertinents,  # Les chunks utilisés (transparence)
        "modele": NOM_MODELE,          # Nom du modèle utilisé
        "tokens_utilises": getattr(reponse_api.usage, "total_tokens", 0)
        # getattr avec default 0 car Ollama ne retourne pas toujours les tokens
    }
```

### `app/main.py` — Point d'entrée FastAPI

```python
# C'est le fichier principal de l'application.
# Il définit toutes les routes (endpoints) de l'API.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import shutil        # Pour copier des fichiers
import os
from pathlib import Path

from .ingestion import charger_document
from .embeddings import indexer_documents
from .chat import repondre_a_question

load_dotenv()

# Créer l'application FastAPI avec des métadonnées
app = FastAPI(
    title="Chatbot FAQ",
    description="API de chatbot FAQ basé sur RAG avec ChromaDB, BGE-M3 et Mistral (Ollama)",
    version="1.0.0"
)

# CORS : Cross-Origin Resource Sharing
# Nécessaire pour que ton frontend (ou Flutter) puisse appeler l'API
# depuis un domaine différent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],         # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],
)

# Dossier où stocker les fichiers uploadés
DOSSIER_UPLOAD = Path("data")
DOSSIER_UPLOAD.mkdir(exist_ok=True)  # Créer le dossier s'il n'existe pas


# Modèles Pydantic pour la validation des données

class QuestionRequest(BaseModel):
    """Modèle pour une question de l'utilisateur"""
    question: str = Field(
        ...,  # "..." signifie "obligatoire" en Pydantic
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
        "message": "API Chatbot FAQ prête à recevoir des questions"
    }


# ROUTE 2 : Upload et indexation d'un document
@app.post("/document/upload", tags=["Documents"])
async def uploader_document(fichier: UploadFile = File(...)):
    """
    Upload un fichier (.txt ou .pdf) et l'indexe dans ChromaDB.
    C'est la phase d'initialisation du RAG.

    Args:
        fichier: Le fichier uploadé via multipart/form-data
    """
    # Vérifier l'extension du fichier
    extensions_autorisees = [".txt", ".pdf", ".md"]
    extension = Path(fichier.filename).suffix.lower()

    if extension not in extensions_autorisees:
        # HTTPException retourne automatiquement une réponse d'erreur JSON
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés : {extensions_autorisees}"
        )

    # Sauvegarder le fichier uploadé sur le disque
    chemin_fichier = DOSSIER_UPLOAD / fichier.filename
    with open(chemin_fichier, "wb") as buffer:
        # shutil.copyfileobj copie les données du fichier uploadé vers le buffer
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
```

### `app/__init__.py`

```python
# Ce fichier (vide ou avec des imports) indique à Python
# que le dossier "app" est un package importable.
# Sans lui, "from app.chat import ..." ne fonctionnerait pas.
```

---

## 1.9 Lancer le projet

### Installation

```bash
# 1. Installer Ollama
# Télécharge l'installeur depuis https://ollama.com et installe-le.
# Puis dans un terminal :
ollama pull mistral   # Télécharge Mistral 7B (~4.1GB, une seule fois)

# 2. Créer un environnement virtuel Python (isoler les dépendances)
python -m venv venv

# 3. Activer l'environnement
# Sur Linux/Mac :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# 4. Installer PyTorch avec support CUDA (pour accélérer BGE-M3 sur ta RTX 4060)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Installer les autres dépendances
pip install -r requirements.txt
# Note : sentence-transformers va télécharger BGE-M3 (~570MB) au premier lancement
```

### Lancement

```bash
# TERMINAL 1 : Lancer Ollama (doit rester ouvert)
ollama serve
# → Serveur Ollama disponible sur http://localhost:11434

# TERMINAL 2 : Lancer FastAPI
uvicorn app.main:app --reload
# → API disponible sur http://localhost:8000

# app.main = module app/main.py
# :app    = la variable "app = FastAPI()" dans ce fichier
# --reload = redémarre automatiquement quand tu modifies le code
```

### Test avec curl

```bash
# 1. Indexer un document
curl -X POST "http://localhost:8000/document/upload" \
  -F "fichier=@data/faq.txt"
# → BGE-M3 génère les embeddings localement, stockés dans ChromaDB

# 2. Poser une question
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont vos horaires ?"}'
# → Mistral génère la réponse localement via Ollama
```

### Interface Swagger automatique

Ouvre http://localhost:8000/docs dans ton navigateur. FastAPI génère automatiquement une interface interactive pour tester tous tes endpoints.

### Vérifier que tout fonctionne

```bash
# Tester Ollama directement
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "Dis bonjour en français"}]
  }'

# Si tu vois une réponse JSON avec du texte en français → Ollama fonctionne ✅
# Si tu vois "connection refused" → lance ollama serve dans un terminal ❌
```

---

## 1.10 Ce qu'il faut mettre sur GitHub

### `.gitignore`

```
# Variables d'environnement
.env

# Environnement virtuel Python
venv/
__pycache__/
*.pyc
*.pyo

# Base ChromaDB (volumineuse, régénérable depuis le document source)
chroma_db/

# Cache des modèles Hugging Face (plusieurs centaines de MB, inutile sur GitHub)
# Ils se retéléchargent automatiquement depuis HuggingFace
.cache/
~/.cache/huggingface/

# Fichiers uploadés (données potentiellement sensibles)
data/*.pdf
data/*.txt

# Fichiers de l'OS
.DS_Store
Thumbs.db
```

> **Pourquoi ne pas committer le cache HuggingFace ?**
> BGE-M3 pèse ~570MB. Le mettre sur GitHub alourdirait inutilement le repo.
> Il se retélécharge automatiquement au premier `uvicorn app.main:app`.

### README.md structure recommandée

```markdown
# Chatbot FAQ — RAG 100% Local avec FastAPI + ChromaDB + Ollama

> ![Démo GIF](demo.gif)  ← Enregistre avec OBS ou ShareX

## ✨ Fonctionnalités
- 🆓 **100% gratuit** : aucune clé API, aucun abonnement
- 🔒 **100% local** : tes données ne quittent jamais ton ordinateur
- 🇫🇷 **Multilingue** : fonctionne en français, anglais, et +100 langues
- 📄 **Multi-format** : PDF et fichiers texte supportés

## Cas d'usage
Automatisation des FAQ pour PME et établissements au Cameroun.
Branchez n'importe quel document PDF/TXT et obtenez un chatbot instantané.

## Architecture RAG
Document PDF/TXT → Découpage en chunks → BGE-M3 (embeddings) → ChromaDB
Question → BGE-M3 (embedding) → Recherche similarité → Mistral 7B → Réponse

## Technologies (toutes gratuites et open source)
- Python 3.11 + FastAPI (API REST)
- ChromaDB (base vectorielle locale)
- BAAI/bge-m3 (embeddings multilingues via sentence-transformers)
- Mistral 7B via Ollama (génération de réponses)

## Prérequis
- Python 3.11+
- Ollama installé (https://ollama.com)
- GPU NVIDIA recommandé (RTX 4060 ou supérieur)

## Installation & Lancement
\```bash
ollama pull mistral
pip install -r requirements.txt
ollama serve &
uvicorn app.main:app --reload
\```

## Exemple d'utilisation
[Screenshot ou GIF de la démo]
```

---

---

# PROJET 2 — Assistant Text-to-SQL avec LangChain et SQLite {#projet-2}

## 2.1 Comprendre le problème

Une PME locale a une base de données avec ses ventes, ses clients, ses stocks. Mais le gérant ne sait pas faire du SQL. Il voudrait juste taper "Combien de clients ont commandé ce mois-ci ?" et obtenir un chiffre.

Le **Text-to-SQL** transforme du langage naturel en requêtes SQL exécutables.

---

## 2.2 Notion fondamentale : LangChain

### Qu'est-ce que LangChain ?

LangChain est un framework qui facilite la construction d'applications avec des LLMs. Au lieu d'écrire tout le code d'orchestration toi-même, LangChain fournit des blocs réutilisables :

- **Chains** : Séquences d'opérations (LLM → traitement → LLM)
- **Agents** : LLMs qui décident quels outils utiliser
- **Memory** : Gestion de l'historique de conversation
- **Tools** : Connexions à des services externes (bases de données, APIs)
- **Retrievers** : Abstractions pour la recherche de documents

### Le concept de "Chain"

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Une chain = prompt | modèle | parser
# L'opérateur | est le "pipe" : la sortie de gauche devient l'entrée de droite

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()  # Convertit la réponse du LLM en string simple

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un expert SQL. Génère uniquement du SQL valide."),
    ("human", "Schéma de la base :\n{schema}\n\nQuestion : {question}")
])

# La chain complète
chain = prompt | llm | parser

# Utilisation
resultat = chain.invoke({
    "schema": "TABLE clients (id, nom, email, date_inscription)",
    "question": "Combien de clients se sont inscrits ce mois-ci ?"
})
# → "SELECT COUNT(*) FROM clients WHERE strftime('%Y-%m', date_inscription) = strftime('%Y-%m', 'now')"
```

---

## 2.3 Notion fondamentale : SQLite

### Qu'est-ce que SQLite ?

SQLite est une base de données relationnelle qui stocke tout dans **un seul fichier** sur le disque. Parfaite pour les projets de démonstration, les petites applications, les prototypes.

Avantages :
- Pas de serveur à installer
- Un seul fichier `.db`
- Supporte presque tout le SQL standard
- Intégrée nativement dans Python (`import sqlite3`)

### Opérations de base

```python
import sqlite3

# 1. Connexion (crée le fichier s'il n'existe pas)
conn = sqlite3.connect("ma_base.db")

# 2. Curseur : objet qui exécute les requêtes
cursor = conn.cursor()

# 3. Créer une table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ventes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        produit TEXT NOT NULL,
        quantite INTEGER,
        prix_unitaire REAL,
        date_vente TEXT,
        client_id INTEGER
    )
""")

# 4. Insérer des données
cursor.execute(
    "INSERT INTO ventes (produit, quantite, prix_unitaire, date_vente, client_id) VALUES (?, ?, ?, ?, ?)",
    ("Paracétamol 500mg", 3, 500.0, "2024-01-15", 1)
    # Les "?" sont des paramètres : TOUJOURS utiliser ça au lieu de f-strings
    # pour éviter les injections SQL
)

# 5. Commit = sauvegarder les changements
conn.commit()

# 6. Lire des données
cursor.execute("SELECT * FROM ventes WHERE date_vente > '2024-01-01'")
lignes = cursor.fetchall()  # Liste de tuples
for ligne in lignes:
    print(ligne)  # (1, "Paracétamol 500mg", 3, 500.0, "2024-01-15", 1)

# 7. Toujours fermer la connexion
conn.close()
```

### Pourquoi les "?" au lieu des f-strings ?

```python
# DANGEREUX - Injection SQL possible !
nom_client = "Robert'; DROP TABLE clients; --"
cursor.execute(f"SELECT * FROM clients WHERE nom = '{nom_client}'")
# Exécute : SELECT * FROM clients WHERE nom = 'Robert'; DROP TABLE clients; --'
# SUPPRIME TOUTE LA TABLE !

# SÉCURISÉ - Les paramètres sont échappés automatiquement
cursor.execute("SELECT * FROM clients WHERE nom = ?", (nom_client,))
# SQLite échappe les caractères dangereux
```

---

## 2.4 Structure du projet

```
text-to-sql/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI
│   ├── database.py      # Gestion SQLite + données de démo
│   ├── sql_agent.py     # Logique Text-to-SQL avec LangChain
│   └── schema.py        # Extraction du schéma de la base
│
├── data/
│   └── boutique.db      # Créé automatiquement
│
├── .env
└── requirements.txt
```

---

## 2.5 Code complet annoté

### `app/database.py` — Gestion de la base SQLite

```python
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import random

DB_PATH = "data/boutique.db"

def creer_base_demo():
    """
    Crée une base de données SQLite avec des données fictives réalistes
    simulant une boutique camerounaise.
    """
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Créer les tables
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            ville TEXT,
            telephone TEXT,
            date_inscription TEXT
        );

        CREATE TABLE IF NOT EXISTS produits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            categorie TEXT,
            prix REAL,
            stock INTEGER
        );

        CREATE TABLE IF NOT EXISTS commandes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            date_commande TEXT,
            total REAL,
            statut TEXT,
            FOREIGN KEY (client_id) REFERENCES clients(id)
        );

        CREATE TABLE IF NOT EXISTS lignes_commande (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commande_id INTEGER,
            produit_id INTEGER,
            quantite INTEGER,
            prix_unitaire REAL,
            FOREIGN KEY (commande_id) REFERENCES commandes(id),
            FOREIGN KEY (produit_id) REFERENCES produits(id)
        );
    """)

    # Insérer des données de démo si les tables sont vides
    if cursor.execute("SELECT COUNT(*) FROM clients").fetchone()[0] == 0:
        clients = [
            ("Mbarga Jean", "Yaoundé", "+237690001111", "2023-06-15"),
            ("Nkomo Cécile", "Douala", "+237670002222", "2023-08-20"),
            ("Foka Pierre", "Bafoussam", "+237655003333", "2023-09-01"),
            ("Abena Marie", "Yaoundé", "+237699004444", "2024-01-10"),
            ("Tamba Robert", "Garoua", "+237688005555", "2024-02-14"),
        ]
        cursor.executemany(
            "INSERT INTO clients (nom, ville, telephone, date_inscription) VALUES (?, ?, ?, ?)",
            clients
        )

        produits = [
            ("Téléphone Samsung A14", "Électronique", 120000, 25),
            ("Câble USB-C", "Accessoires", 3500, 150),
            ("Sachet de riz 25kg", "Alimentation", 18000, 80),
            ("Huile de palme 5L", "Alimentation", 6500, 60),
            ("Cartouche imprimante HP", "Bureautique", 15000, 30),
        ]
        cursor.executemany(
            "INSERT INTO produits (nom, categorie, prix, stock) VALUES (?, ?, ?, ?)",
            produits
        )

        # Générer des commandes aléatoires sur les 90 derniers jours
        for i in range(20):
            date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
            client_id = random.randint(1, 5)
            total = random.uniform(5000, 150000)
            statut = random.choice(["livré", "en cours", "annulé"])
            cursor.execute(
                "INSERT INTO commandes (client_id, date_commande, total, statut) VALUES (?, ?, ?, ?)",
                (client_id, date, round(total, 2), statut)
            )

    conn.commit()
    conn.close()
    print("✅ Base de données de démo créée !")


def executer_requete(sql: str) -> dict:
    """
    Exécute une requête SQL et retourne les résultats formatés.

    Args:
        sql: La requête SQL à exécuter

    Returns:
        Dictionnaire avec colonnes, lignes et nombre de résultats
    """
    conn = sqlite3.connect(DB_PATH)

    # row_factory permet d'accéder aux colonnes par nom : row["nom"] au lieu de row[0]
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        lignes = cursor.fetchall()

        if lignes:
            colonnes = list(lignes[0].keys())
            donnees = [dict(ligne) for ligne in lignes]
        else:
            colonnes = []
            donnees = []

        return {
            "succes": True,
            "colonnes": colonnes,
            "donnees": donnees,
            "nombre_resultats": len(donnees)
        }

    except sqlite3.Error as e:
        return {
            "succes": False,
            "erreur": str(e),
            "donnees": [],
            "nombre_resultats": 0
        }
    finally:
        conn.close()
```

### `app/schema.py` — Extraction du schéma

```python
import sqlite3
from .database import DB_PATH

def obtenir_schema_complet() -> str:
    """
    Génère une description textuelle complète du schéma de la base.
    Cette description sera envoyée au LLM pour qu'il comprenne la structure.

    Returns:
        Description lisible du schéma (tables, colonnes, types, exemples)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Récupérer la liste de toutes les tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = ["BASE DE DONNÉES : Boutique (SQLite)\n"]

    for table in tables:
        # Récupérer les informations sur les colonnes de chaque table
        cursor.execute(f"PRAGMA table_info({table})")
        colonnes = cursor.fetchall()

        # Récupérer 2 exemples de données pour aider le LLM
        cursor.execute(f"SELECT * FROM {table} LIMIT 2")
        exemples = cursor.fetchall()

        schema_parts.append(f"\nTABLE {table.upper()}:")
        for col in colonnes:
            # col = (cid, name, type, notnull, default, pk)
            pk = " [CLÉ PRIMAIRE]" if col[5] else ""
            null = " NOT NULL" if col[3] else ""
            schema_parts.append(f"  - {col[1]} : {col[2]}{null}{pk}")

        if exemples:
            schema_parts.append(f"  Exemples : {exemples}")

    conn.close()
    return "\n".join(schema_parts)
```

### `app/sql_agent.py` — Logique Text-to-SQL

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import os

from .schema import obtenir_schema_complet
from .database import executer_requete

load_dotenv()


def nettoyer_sql(texte_brut: str) -> str:
    """
    Extrait la requête SQL pure depuis la réponse du LLM.
    Le LLM peut retourner du markdown avec ```sql ... ``` qu'il faut enlever.

    Args:
        texte_brut: La réponse brute du LLM

    Returns:
        La requête SQL nettoyée
    """
    # Supprimer les blocs markdown ```sql ... ```
    texte = re.sub(r"```sql\s*", "", texte_brut)
    texte = re.sub(r"```\s*", "", texte)

    # Supprimer les espaces en début/fin
    return texte.strip()


def creer_chain_sql():
    """
    Crée la chaîne LangChain pour la génération de SQL.

    Returns:
        Une chain LangChain exécutable
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0  # 0 = déterministe : on veut toujours le même SQL pour la même question
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un expert SQL spécialisé en SQLite.

RÈGLES ABSOLUES :
1. Génère UNIQUEMENT la requête SQL, sans explication, sans markdown.
2. Utilise UNIQUEMENT les tables et colonnes existantes dans le schéma fourni.
3. Pour les dates, utilise la fonction SQLite strftime('%Y-%m-%d', ...).
4. Pour "ce mois-ci" → strftime('%Y-%m', date_col) = strftime('%Y-%m', 'now')
5. Pour "aujourd'hui" → date(date_col) = date('now')
6. Termine toujours la requête par un point-virgule.
7. N'utilise jamais DROP, DELETE, UPDATE, INSERT — lecture seulement.

SCHÉMA DE LA BASE :
{schema}"""),
        ("human", "{question}")
    ])

    # La chain : prompt → LLM → parser (string)
    return prompt | llm | StrOutputParser()


def creer_chain_interpretation():
    """
    Crée une chain pour interpréter les résultats SQL en langage naturel.
    L'utilisateur voit une phrase claire, pas juste des données brutes.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un analyste business. Tu interprètes des résultats de requêtes SQL
en phrases claires en français. Sois précis, mentionne les chiffres clés.
Si les résultats sont vides, explique que aucune donnée ne correspond."""),
        ("human", """Question posée : {question}
Requête SQL exécutée : {sql}
Résultats : {resultats}

Donne une réponse claire en une ou deux phrases.""")
    ])

    return prompt | llm | StrOutputParser()


def questionner_base(question: str) -> dict:
    """
    Pipeline complet : question → SQL → exécution → interprétation.

    Args:
        question: Question en langage naturel

    Returns:
        Dictionnaire avec la requête SQL, les données brutes et l'interprétation
    """
    schema = obtenir_schema_complet()

    # ÉTAPE 1 : Générer le SQL
    chain_sql = creer_chain_sql()
    sql_brut = chain_sql.invoke({"schema": schema, "question": question})
    sql_propre = nettoyer_sql(sql_brut)

    # ÉTAPE 2 : Exécuter le SQL
    resultats = executer_requete(sql_propre)

    if not resultats["succes"]:
        return {
            "question": question,
            "sql": sql_propre,
            "erreur": resultats["erreur"],
            "donnees": [],
            "interpretation": f"Erreur lors de l'exécution : {resultats['erreur']}"
        }

    # ÉTAPE 3 : Interpréter les résultats en langage naturel
    chain_interpretation = creer_chain_interpretation()
    interpretation = chain_interpretation.invoke({
        "question": question,
        "sql": sql_propre,
        "resultats": str(resultats["donnees"][:10])  # Limite à 10 pour ne pas dépasser le contexte
    })

    return {
        "question": question,
        "sql": sql_propre,
        "donnees": resultats["donnees"],
        "nombre_resultats": resultats["nombre_resultats"],
        "interpretation": interpretation
    }
```

---

---

# PROJET 3 — App Flutter de Chatbot IA {#projet-3}

## 3.1 Comprendre le problème

Tu as un backend Python (Projet 1 ou 2). Tu veux une vraie application mobile pour le démontrer à des clients. Flutter permet de créer une app iOS et Android depuis un seul code.

---

## 3.2 Notion fondamentale : Flutter et Dart

### Qu'est-ce que Flutter ?

Flutter est un framework créé par Google pour construire des applications multiplateformes (iOS, Android, Web, Desktop) depuis un seul code source écrit en **Dart**.

### Les concepts clés de Flutter

#### Les Widgets

Tout dans Flutter est un Widget. Un widget est un bloc d'interface (bouton, texte, colonne, ligne). Ils s'emboîtent comme des Lego.

```dart
// Un widget simple
Text("Bonjour")

// Des widgets imbriqués
Column(             // Colonne verticale
  children: [
    Text("Titre"),  // Premier enfant
    SizedBox(height: 8),  // Espace
    ElevatedButton(       // Bouton
      onPressed: () => print("cliqué"),
      child: Text("Cliquer"),
    ),
  ],
)
```

#### Stateless vs Stateful Widgets

```dart
// StatelessWidget : ne change jamais une fois affiché
class TitreApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Text("Mon Chatbot", style: TextStyle(fontSize: 24));
  }
}

// StatefulWidget : peut changer (messages qui arrivent, chargement...)
class PageChat extends StatefulWidget {
  @override
  _PageChatState createState() => _PageChatState();
}

class _PageChatState extends State<PageChat> {
  List<String> messages = [];  // L'état : les messages du chat

  void ajouterMessage(String message) {
    setState(() {                // setState() = "reconstruire l'interface"
      messages.add(message);
    });
  }

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: messages.length,
      itemBuilder: (context, index) => Text(messages[index]),
    );
  }
}
```

#### async/await en Dart

```dart
// En Dart (comme en JavaScript), les opérations réseau sont asynchrones
// async = cette fonction peut attendre
// await = attendre que cette opération se termine

Future<String> envoyerQuestion(String question) async {
  // http.post retourne un Future (promesse de résultat)
  final reponse = await http.post(
    Uri.parse("http://localhost:8000/chat"),
    headers: {"Content-Type": "application/json"},
    body: jsonEncode({"question": question}),
  );

  if (reponse.statusCode == 200) {
    final donnees = jsonDecode(reponse.body);
    return donnees["reponse"];
  } else {
    throw Exception("Erreur API : ${reponse.statusCode}");
  }
}
```

---

## 3.3 Notion fondamentale : sqflite (SQLite pour Flutter)

sqflite est un plugin Flutter qui donne accès à SQLite sur mobile. Permet de stocker l'historique des conversations localement, sans internet.

```dart
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class BaseDonneesLocale {
  static Database? _db;

  // Singleton : une seule instance de la base
  static Future<Database> get db async {
    if (_db != null) return _db!;
    _db = await _initialiser();
    return _db!;
  }

  static Future<Database> _initialiser() async {
    // Chemin vers le fichier de base de données sur le téléphone
    String chemin = join(await getDatabasesPath(), "chatbot.db");

    return await openDatabase(
      chemin,
      version: 1,
      // onCreate est appelé uniquement à la première ouverture
      onCreate: (db, version) async {
        await db.execute("""
          CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,           -- 'user' ou 'assistant'
            contenu TEXT NOT NULL,
            timestamp INTEGER NOT NULL,   -- Unix timestamp
            session_id TEXT NOT NULL      -- ID de la conversation
          )
        """);
      },
    );
  }

  // Sauvegarder un message
  static Future<void> sauvegarderMessage({
    required String role,
    required String contenu,
    required String sessionId,
  }) async {
    final database = await db;
    await database.insert("messages", {
      "role": role,
      "contenu": contenu,
      "timestamp": DateTime.now().millisecondsSinceEpoch,
      "session_id": sessionId,
    });
  }

  // Récupérer tous les messages d'une session
  static Future<List<Map<String, dynamic>>> getMessages(String sessionId) async {
    final database = await db;
    return await database.query(
      "messages",
      where: "session_id = ?",
      whereArgs: [sessionId],
      orderBy: "timestamp ASC",
    );
  }
}
```

---

## 3.4 Code Flutter complet

### `pubspec.yaml` — Dépendances

```yaml
name: chatbot_faq
description: Application de chatbot FAQ avec IA

environment:
  sdk: ">=3.0.0 <4.0.0"

dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0          # Requêtes HTTP vers l'API
  sqflite: ^2.3.2       # SQLite local
  path: ^1.9.0          # Manipulation de chemins
  uuid: ^4.3.3          # Générer des IDs uniques pour les sessions
  intl: ^0.19.0         # Formatage des dates

dev_dependencies:
  flutter_test:
    sdk: flutter
```

### `lib/models/message.dart` — Modèle de message

```dart
// Un modèle de données représente un objet métier dans l'application

class Message {
  final String id;
  final String role;      // "user" ou "assistant"
  final String contenu;
  final DateTime timestamp;

  Message({
    required this.id,
    required this.role,
    required this.contenu,
    required this.timestamp,
  });

  // Constructeur depuis une Map (pour SQLite)
  factory Message.fromMap(Map<String, dynamic> map) {
    return Message(
      id: map["id"].toString(),
      role: map["role"],
      contenu: map["contenu"],
      timestamp: DateTime.fromMillisecondsSinceEpoch(map["timestamp"]),
    );
  }

  // Conversion en Map (pour SQLite)
  Map<String, dynamic> toMap() {
    return {
      "role": role,
      "contenu": contenu,
      "timestamp": timestamp.millisecondsSinceEpoch,
    };
  }

  bool get estUtilisateur => role == "user";
}
```

### `lib/services/api_service.dart` — Connexion au backend

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/message.dart';

class ApiService {
  // URL de ton backend FastAPI
  // En développement : ton IP locale (pas localhost depuis le téléphone !)
  static const String baseUrl = "http://192.168.1.X:8000";
  // Sur Render.com : "https://ton-projet.onrender.com"

  /// Envoie une question au chatbot et retourne la réponse
  static Future<Map<String, dynamic>> envoyerQuestion({
    required String question,
    List<Message> historique = const [],
  }) async {
    // Convertir l'historique en format attendu par l'API
    final historiqueApi = historique.map((msg) => {
      "role": msg.role,
      "content": msg.contenu,
    }).toList();

    try {
      final reponse = await http.post(
        Uri.parse("$baseUrl/chat"),
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: jsonEncode({
          "question": question,
          "historique": historiqueApi,
        }),
      ).timeout(
        const Duration(seconds: 30),  // Timeout : 30 secondes max
        onTimeout: () => throw Exception("Le serveur met trop de temps à répondre"),
      );

      if (reponse.statusCode == 200) {
        return jsonDecode(reponse.body);
      } else {
        throw Exception("Erreur serveur : ${reponse.statusCode}");
      }
    } on Exception catch (e) {
      throw Exception("Impossible de contacter le serveur : $e");
    }
  }
}
```

### `lib/screens/chat_screen.dart` — Interface principale

```dart
import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';
import '../models/message.dart';
import '../services/api_service.dart';
import '../services/database_service.dart';
import '../widgets/message_bubble.dart';

class ChatScreen extends StatefulWidget {
  final String sessionId;
  const ChatScreen({super.key, required this.sessionId});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // Contrôleur pour le champ de texte
  final TextEditingController _controleur = TextEditingController();

  // Contrôleur pour le scroll automatique vers le bas
  final ScrollController _scrollControleur = ScrollController();

  // Liste des messages affichés
  List<Message> _messages = [];

  // Indicateur de chargement (IA en train de répondre)
  bool _enChargement = false;

  @override
  void initState() {
    super.initState();
    _chargerHistorique();  // Charger les anciens messages au démarrage
  }

  Future<void> _chargerHistorique() async {
    final messages = await DatabaseService.getMessages(widget.sessionId);
    setState(() {
      _messages = messages.map((m) => Message.fromMap(m)).toList();
    });
    _scrollerVersLeBas();
  }

  Future<void> _envoyerMessage() async {
    final texte = _controleur.text.trim();
    if (texte.isEmpty || _enChargement) return;

    // Créer le message utilisateur
    final messageUser = Message(
      id: const Uuid().v4(),
      role: "user",
      contenu: texte,
      timestamp: DateTime.now(),
    );

    // Afficher immédiatement le message de l'utilisateur
    setState(() {
      _messages.add(messageUser);
      _enChargement = true;
    });

    _controleur.clear();
    _scrollerVersLeBas();

    // Sauvegarder en local
    await DatabaseService.sauvegarderMessage(
      role: "user",
      contenu: texte,
      sessionId: widget.sessionId,
    );

    try {
      // Appeler l'API
      final resultat = await ApiService.envoyerQuestion(
        question: texte,
        historique: _messages.where((m) => m != messageUser).toList(),
      );

      // Créer le message de l'IA
      final messageIA = Message(
        id: const Uuid().v4(),
        role: "assistant",
        contenu: resultat["reponse"],
        timestamp: DateTime.now(),
      );

      setState(() {
        _messages.add(messageIA);
        _enChargement = false;
      });

      // Sauvegarder la réponse en local
      await DatabaseService.sauvegarderMessage(
        role: "assistant",
        contenu: resultat["reponse"],
        sessionId: widget.sessionId,
      );

    } catch (e) {
      setState(() {
        _messages.add(Message(
          id: const Uuid().v4(),
          role: "assistant",
          contenu: "Erreur : $e",
          timestamp: DateTime.now(),
        ));
        _enChargement = false;
      });
    }

    _scrollerVersLeBas();
  }

  void _scrollerVersLeBas() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollControleur.hasClients) {
        _scrollControleur.animateTo(
          _scrollControleur.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Assistant FAQ"),
        backgroundColor: const Color(0xFF1A73E8),
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Zone de messages (prend tout l'espace disponible)
          Expanded(
            child: _messages.isEmpty
              ? const Center(
                  child: Text(
                    "Posez votre première question !",
                    style: TextStyle(color: Colors.grey, fontSize: 16),
                  ),
                )
              : ListView.builder(
                  controller: _scrollControleur,
                  padding: const EdgeInsets.all(12),
                  itemCount: _messages.length + (_enChargement ? 1 : 0),
                  itemBuilder: (context, index) {
                    // Afficher l'indicateur "en train d'écrire..."
                    if (index == _messages.length) {
                      return const TypingIndicator();
                    }
                    return MessageBubble(message: _messages[index]);
                  },
                ),
          ),

          // Barre de saisie
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withOpacity(0.2),
                  spreadRadius: 1,
                  blurRadius: 4,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: Row(
              children: [
                // Champ de texte
                Expanded(
                  child: TextField(
                    controller: _controleur,
                    decoration: InputDecoration(
                      hintText: "Posez votre question...",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(24),
                        borderSide: BorderSide.none,
                      ),
                      filled: true,
                      fillColor: Colors.grey[100],
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 10,
                      ),
                    ),
                    maxLines: null,  // S'agrandit si le texte est long
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _envoyerMessage(),
                  ),
                ),
                const SizedBox(width: 8),

                // Bouton d'envoi
                FloatingActionButton.small(
                  onPressed: _enChargement ? null : _envoyerMessage,
                  backgroundColor: const Color(0xFF1A73E8),
                  child: _enChargement
                    ? const SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(
                          color: Colors.white, strokeWidth: 2,
                        ),
                      )
                    : const Icon(Icons.send, color: Colors.white),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    // Toujours libérer les ressources pour éviter les fuites mémoire
    _controleur.dispose();
    _scrollControleur.dispose();
    super.dispose();
  }
}

// Widget pour l'indicateur "en train d'écrire..."
class TypingIndicator extends StatefulWidget {
  const TypingIndicator({super.key});

  @override
  State<TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<TypingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat();
  }

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8, right: 60),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: BorderRadius.circular(16),
        ),
        child: AnimatedBuilder(
          animation: _controller,
          builder: (context, child) {
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                return Container(
                  margin: const EdgeInsets.symmetric(horizontal: 2),
                  width: 8, height: 8,
                  decoration: BoxDecoration(
                    color: Colors.grey[600]!.withOpacity(
                      ((_controller.value * 3 - i) % 1).abs() < 0.5 ? 1.0 : 0.4
                    ),
                    shape: BoxShape.circle,
                  ),
                );
              }),
            );
          },
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
```

---

---

# PROJET 4 — Agent d'analyse de documents avec pgvector {#projet-4}

## 4.1 Pourquoi pgvector au lieu de ChromaDB ?

ChromaDB est parfait pour commencer, mais en production, les entreprises utilisent déjà PostgreSQL. pgvector ajoute des capacités vectorielles directement dans PostgreSQL, sans base supplémentaire à maintenir.

Avantages de pgvector :
- Tout dans une seule base (données + vecteurs)
- Transactions ACID (sécurité des données)
- SQL standard + requêtes vectorielles dans la même query
- Scalable pour des millions de documents

---

## 4.2 Notion fondamentale : PostgreSQL

### Différences avec SQLite

| SQLite | PostgreSQL |
|--------|-----------|
| Fichier local | Serveur dédié |
| 1 utilisateur à la fois | Milliers d'utilisateurs simultanés |
| Pas de types avancés | JSON, Arrays, Types personnalisés |
| Pas d'extensions | Extensions (dont pgvector) |
| Projets solo/démo | Production, entreprises |

### Connexion avec psycopg2

```python
import psycopg2
from psycopg2.extras import RealDictCursor

# Connexion au serveur PostgreSQL
conn = psycopg2.connect(
    host="localhost",      # Serveur
    port=5432,             # Port par défaut de PostgreSQL
    database="chatbot_db", # Nom de la base
    user="postgres",       # Utilisateur
    password="monmotdepasse"
)

# RealDictCursor : les résultats sont des dictionnaires (row["colonne"])
cursor = conn.cursor(cursor_factory=RealDictCursor)

cursor.execute("SELECT * FROM documents WHERE id = %s", (1,))
# Note : PostgreSQL utilise %s comme placeholder (pas ? comme SQLite)
```

### Installation de pgvector

```bash
# Sur Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# Dans PostgreSQL (connexion psql)
CREATE EXTENSION IF NOT EXISTS vector;

# Créer une table avec une colonne vectorielle
CREATE TABLE documents_chunks (
    id SERIAL PRIMARY KEY,
    contenu TEXT NOT NULL,
    source TEXT,
    page_num INTEGER,
    embedding vector(1536)  -- 1536 dimensions pour text-embedding-3-small
);

# Créer un index pour accélérer les recherches (IVFFLAT = index approximatif)
CREATE INDEX ON documents_chunks USING ivfflat (embedding vector_cosine_ops);
```

### Requête de similarité avec pgvector

```sql
-- Trouve les 5 chunks les plus proches du vecteur donné
-- <=> = opérateur de distance cosinus dans pgvector
SELECT contenu, source, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarite
FROM documents_chunks
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

---

## 4.3 Notion fondamentale : LangChain avec PGVector

LangChain a un connecteur intégré pour pgvector qui gère tout automatiquement.

```python
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

# Créer le vectorstore (se connecte à PostgreSQL)
vectorstore = PGVector(
    connection="postgresql://user:password@localhost:5432/chatbot_db",
    collection_name="documents_chunks",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# Ajouter des documents (génère et stocke les embeddings automatiquement)
vectorstore.add_texts(
    texts=["Notre école est ouverte de 8h à 18h", "Les inscriptions coûtent 50000 FCFA"],
    metadatas=[{"source": "reglement.pdf", "page": 1}, {"source": "reglement.pdf", "page": 2}]
)

# Recherche par similarité
resultats = vectorstore.similarity_search("horaires d'ouverture", k=3)
for doc in resultats:
    print(doc.page_content)
    print(doc.metadata)
```

---

## 4.4 Code clé du projet 4

### `app/vectorstore.py`

```python
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# RecursiveCharacterTextSplitter : plus intelligent que le découpage simple
# Il essaie de couper aux paragraphes, puis aux phrases, puis aux mots
decoupeur = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]  # Ordre de priorité pour couper
)

def obtenir_vectorstore():
    """Retourne le vectorstore connecté à PostgreSQL."""
    return PGVector(
        connection=os.getenv("DATABASE_URL"),
        collection_name="documents",
        embeddings=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
    )

def indexer_pdf(chemin_pdf: str, nom_source: str):
    """
    Charge un PDF, le découpe et l'indexe dans pgvector.
    PyPDFLoader est le chargeur LangChain officiel pour PDF.
    """
    # Charger le PDF (retourne une liste de Documents LangChain, un par page)
    chargeur = PyPDFLoader(chemin_pdf)
    pages = chargeur.load()

    # Découper en chunks
    chunks = decoupeur.split_documents(pages)

    # Ajouter des métadonnées personnalisées
    for chunk in chunks:
        chunk.metadata["nom_source"] = nom_source

    # Indexer (génère les embeddings + stocke dans PostgreSQL)
    vs = obtenir_vectorstore()
    vs.add_documents(chunks)

    return len(chunks)

def rechercher(question: str, k: int = 4) -> list:
    """Recherche les chunks les plus pertinents."""
    vs = obtenir_vectorstore()
    return vs.similarity_search_with_score(question, k=k)
```

### `app/agent.py` — Agent avec mémoire de la source

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import os
from .vectorstore import obtenir_vectorstore

def creer_agent_qa():
    """
    RetrievalQAWithSourcesChain = Chain RAG qui cite automatiquement les sources.
    Très utile pour montrer d'où vient chaque information.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    vs = obtenir_vectorstore()

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = tout mettre dans un seul prompt (simple et efficace)
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    return chain

def analyser_document(question: str) -> dict:
    """Répond à une question en citant les sources."""
    chain = creer_agent_qa()
    resultat = chain.invoke({"question": question})

    # Extraire les pages sources pour la transparence
    sources = []
    for doc in resultat.get("source_documents", []):
        sources.append({
            "contenu": doc.page_content[:200] + "...",
            "page": doc.metadata.get("page", "?"),
            "source": doc.metadata.get("nom_source", "?")
        })

    return {
        "reponse": resultat["answer"],
        "sources": sources
    }
```

---

---

# PROJET 5 — Système Multi-Agents avec LangGraph {#projet-5}

## 5.1 Pourquoi les multi-agents ?

Un seul LLM peut faire beaucoup, mais il a des limites. Un système multi-agents divise une tâche complexe entre plusieurs agents spécialisés, coordonnés par un agent superviseur.

Exemple : "Analyse les ventes de ce mois et rédige un rapport"
- **Agent Analyst** : Interroge la base de données
- **Agent Researcher** : Cherche des infos contextuelles sur le web
- **Agent Writer** : Rédige le rapport final
- **Agent Supervisor** : Coordonne tout

---

## 5.2 Notion fondamentale : LangGraph

LangGraph modélise un workflow d'agents comme un **graphe d'états** :
- Les **nœuds** sont des agents ou des fonctions
- Les **arêtes** définissent les transitions (qui appelle qui)
- L'**état** est partagé entre tous les nœuds

```
START → Supervisor → Analyst → Supervisor → Writer → END
                  ↘ Researcher ↗
```

### Concepts de base

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Définir l'état partagé
class EtatAgent(TypedDict):
    messages: Annotated[list, operator.add]  # Liste de messages (s'accumule)
    prochaine_etape: str                      # Quel agent appeler ensuite
    resultat_final: str                       # Résultat à retourner à l'utilisateur

# 2. Définir les nœuds (agents)
def agent_analyste(etat: EtatAgent) -> EtatAgent:
    """Interroge la base de données."""
    question = etat["messages"][-1]
    # ... logique SQL ...
    return {
        "messages": [f"Résultat SQL : {donnees}"],
        "prochaine_etape": "writer"
    }

def agent_writer(etat: EtatAgent) -> EtatAgent:
    """Rédige la réponse finale."""
    contexte = "\n".join(etat["messages"])
    # ... appel LLM pour rédiger ...
    return {
        "messages": [reponse],
        "prochaine_etape": "end",
        "resultat_final": reponse
    }

def router(etat: EtatAgent) -> str:
    """Décide quel nœud appeler en fonction de l'état."""
    return etat["prochaine_etape"]

# 3. Construire le graphe
graphe = StateGraph(EtatAgent)

# Ajouter les nœuds
graphe.add_node("analyste", agent_analyste)
graphe.add_node("writer", agent_writer)

# Ajouter les arêtes conditionnelles
graphe.add_conditional_edges(
    "analyste",
    router,
    {"writer": "writer", "end": END}
)
graphe.add_conditional_edges(
    "writer",
    router,
    {"end": END}
)

# Point d'entrée
graphe.set_entry_point("analyste")

# Compiler le graphe
app_agent = graphe.compile()
```

---

## 5.3 Code clé du Projet 5

### `app/agents/supervisor.py`

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

AGENTS_DISPONIBLES = ["analyst", "researcher", "writer", "FINISH"]

prompt_supervisor = ChatPromptTemplate.from_messages([
    ("system", f"""Tu es un superviseur qui coordonne une équipe d'agents IA.
Agents disponibles : {AGENTS_DISPONIBLES}

- analyst : interroge la base de données SQL
- researcher : fait des recherches web
- writer : rédige des réponses ou rapports
- FINISH : quand le travail est terminé

Analyse la conversation et décide quel agent appeler.
Réponds UNIQUEMENT en JSON : {{"prochaine_action": "nom_agent", "raison": "pourquoi"}}"""),
    ("human", "Conversation : {messages}\n\nQuel agent appeler ?")
])

chaine_supervisor = prompt_supervisor | llm | JsonOutputParser()

def supervisor(etat):
    """Agent superviseur : décide qui appeler ensuite."""
    decision = chaine_supervisor.invoke({"messages": str(etat["messages"])})
    return {
        "prochaine_etape": decision["prochaine_action"],
        "messages": [f"[Supervisor] → {decision['prochaine_action']} : {decision['raison']}"]
    }
```

### `app/agents/analyst.py`

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from ..database import executer_requete, obtenir_schema

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

def agent_analyst(etat):
    """Agent qui génère et exécute des requêtes SQL."""
    # Extraire la dernière question de l'état
    derniere_question = [m for m in etat["messages"] if isinstance(m, str) and not m.startswith("[")][-1]

    # Générer le SQL
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Génère du SQL SQLite pour : {obtenir_schema()}. Retourne SEULEMENT le SQL."),
        ("human", derniere_question)
    ])
    sql = (prompt | llm).invoke({}).content.strip()

    # Exécuter
    resultat = executer_requete(sql)

    return {
        "messages": [f"[Analyst] SQL: {sql}\nRésultat: {resultat['donnees'][:5]}"],
        "prochaine_etape": "supervisor"  # Retourner au superviseur après chaque action
    }
```

### `app/graph.py` — Construction du graphe complet

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from .agents.supervisor import supervisor
from .agents.analyst import agent_analyst
from .agents.researcher import agent_researcher
from .agents.writer import agent_writer

class EtatMultiAgent(TypedDict):
    messages: Annotated[list, operator.add]
    prochaine_etape: str
    resultat_final: str

def router(etat: EtatMultiAgent) -> str:
    """Route vers le bon agent selon l'état."""
    etape = etat.get("prochaine_etape", "supervisor")
    if etape == "FINISH":
        return END
    return etape

# Construire le graphe
graphe = StateGraph(EtatMultiAgent)

# Ajouter tous les agents comme nœuds
graphe.add_node("supervisor", supervisor)
graphe.add_node("analyst", agent_analyst)
graphe.add_node("researcher", agent_researcher)
graphe.add_node("writer", agent_writer)

# Toutes les transitions passent par le router
for agent in ["supervisor", "analyst", "researcher", "writer"]:
    graphe.add_conditional_edges(
        agent,
        router,
        {
            "supervisor": "supervisor",
            "analyst": "analyst",
            "researcher": "researcher",
            "writer": "writer",
            END: END
        }
    )

graphe.set_entry_point("supervisor")
app_multiagent = graphe.compile()


def executer_tache(tache: str) -> dict:
    """
    Lance le système multi-agents pour exécuter une tâche.

    Args:
        tache: La tâche à accomplir en langage naturel

    Returns:
        L'état final avec tous les messages et le résultat
    """
    etat_initial = {
        "messages": [tache],
        "prochaine_etape": "supervisor",
        "resultat_final": ""
    }

    # stream() exécute le graphe étape par étape et yield chaque état intermédiaire
    # Permet de voir les agents travailler en temps réel !
    etats = []
    for etat in app_multiagent.stream(etat_initial):
        etats.append(etat)

    return etats[-1]  # Retourner l'état final
```

---

---

# ANNEXE — Bonnes pratiques transversales

## Gestion des erreurs

```python
# Toujours wrapper les appels API dans des try/except
# Les APIs peuvent tomber, avoir des timeouts, retourner des erreurs

from openai import APIError, APITimeoutError, APIConnectionError
import time

def appel_ollama_robuste(messages: list, max_retries: int = 3) -> str:
    """
    Appel Ollama avec retry automatique.
    Avec Ollama local, les erreurs typiques sont :
    - Timeout si le modèle est lent à répondre
    - ConnectionError si Ollama n'est pas lancé
    - Pas de RateLimitError (on est en local !)
    """
    for tentative in range(max_retries):
        try:
            reponse = client_ollama.chat.completions.create(
                model=NOM_MODELE,
                messages=messages,
                temperature=0.1,
            )
            return reponse.choices[0].message.content

        except APIConnectionError:
            # Ollama n'est pas lancé → message clair
            raise RuntimeError(
                "Impossible de contacter Ollama. "
                "Lance 'ollama serve' dans un terminal."
            )

        except APITimeoutError:
            # Le modèle met trop de temps (rare sur RTX 4060 avec 7B)
            attente = 2 ** tentative  # Backoff exponentiel : 1s, 2s, 4s
            print(f"⏱️  Timeout Ollama. Tentative {tentative+1}/{max_retries}. Attente {attente}s...")
            time.sleep(attente)

        except APIError as e:
            print(f"❌ Erreur Ollama : {e}")
            time.sleep(1)

    raise Exception("Toutes les tentatives ont échoué. Vérifie qu'Ollama est lancé.")
```

## Variables d'environnement

```python
# Toujours utiliser des variables d'environnement pour la configuration
# Avantage : changer de modèle ou d'URL sans modifier le code

# .env (Projet 1 — stack 100% locale)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=BAAI/bge-m3

# .env (Projets 2-5 avec base de données)
DATABASE_URL=postgresql://user:pass@localhost/db
SECRET_KEY=une_cle_aleatoire_longue

# Python
from dotenv import load_dotenv
import os

load_dotenv()

# Vérification que les variables critiques sont définies
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
modele = os.getenv("OLLAMA_MODEL", "mistral")  # Valeur par défaut si non définie

# Test de connexion Ollama au démarrage
import httpx
try:
    httpx.get(ollama_url.replace("/v1", ""), timeout=2)
    print(f"✅ Ollama accessible sur {ollama_url}")
except Exception:
    print(f"⚠️  Ollama non accessible. Lance 'ollama serve'")
```

## Structure d'un bon README GitHub

```markdown
# Nom du Projet

![Demo GIF](demo.gif)  ← OBLIGATOIRE : un GIF vaut 1000 mots

## 🎯 Cas d'usage
Une phrase qui explique le problème résolu.

## 🏗️ Architecture
[Schéma ou diagramme]

## 🛠️ Stack technique
- Python 3.11 + FastAPI
- OpenAI GPT-4o-mini
- ChromaDB / pgvector

## 🚀 Lancement rapide
\```bash
git clone ...
pip install -r requirements.txt
cp .env.example .env  # Remplir avec ta clé API
uvicorn app.main:app --reload
\```

## 📡 Endpoints API
| Méthode | Route | Description |
|---------|-------|-------------|
| POST | /chat | Poser une question |
| POST | /document/upload | Indexer un document |

## 💡 Améliorations futures
- [ ] Support multilingue
- [ ] Interface web
```

---

*Ce cours couvre les fondamentaux nécessaires pour réaliser les 5 projets. Chaque section peut être approfondie avec la documentation officielle de chaque outil.*
