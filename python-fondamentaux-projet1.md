# Cours Python Fondamentaux — Expliqué avec le Projet 1

> **Comment lire ce cours**
> Chaque concept est expliqué en 3 temps :
> 1. **C'est quoi ?** — La définition simple
> 2. **Comment ça marche ?** — Exemples progressifs
> 3. **Dans notre projet** — Où et pourquoi on l'utilise

---

# TABLE DES MATIÈRES

1. [Les imports et modules](#1-les-imports-et-modules)
2. [Les variables et types de données](#2-les-variables-et-types-de-données)
3. [Les fonctions et def](#3-les-fonctions-et-def)
4. [Les f-strings](#4-les-f-strings)
5. [Les listes et leurs opérations](#5-les-listes-et-leurs-opérations)
6. [Les dictionnaires](#6-les-dictionnaires)
7. [Le typage (annotations de type)](#7-le-typage-annotations-de-type)
8. [Les classes et Pydantic](#8-les-classes-et-pydantic)
9. [Les décorateurs (@)](#9-les-décorateurs-)
10. [async / await](#10-async--await)
11. [Les gestionnaires de contexte (with)](#11-les-gestionnaires-de-contexte-with)
12. [La gestion des erreurs (try/except)](#12-la-gestion-des-erreurs-tryexcept)
13. [Les variables d'environnement et dotenv](#13-les-variables-denvironnement-et-dotenv)
14. [La structure en modules (les points dans les imports)](#14-la-structure-en-modules)
15. [List comprehensions](#15-list-comprehensions)

---

# 1. Les imports et modules

## C'est quoi ?

Python seul ne sait faire que des choses basiques. Les **modules** sont des fichiers Python qui ajoutent des fonctionnalités. Les **imports** permettent d'utiliser ces fonctionnalités dans ton code.

Analogie : Python c'est un téléphone neuf. Les modules c'est les applications que tu installes. L'import c'est ouvrir l'application.

## Comment ça marche ?

```python
# SYNTAXE 1 : importer tout un module
import os           # Le module "os" gère les fichiers et dossiers
import chromadb     # Le module ChromaDB qu'on a installé avec pip

# Utilisation : nomModule.fonctionOuVariable
taille = os.path.getsize("faq.txt")  # os = module, path = sous-module, getsize = fonction

# SYNTAXE 2 : importer seulement ce dont on a besoin
from pathlib import Path     # Depuis "pathlib", prendre uniquement "Path"
from fastapi import FastAPI  # Depuis "fastapi", prendre uniquement "FastAPI"

# Utilisation directe, sans préfixe
chemin = Path("data/faq.txt")   # Pas besoin d'écrire pathlib.Path
app = FastAPI()                  # Pas besoin d'écrire fastapi.FastAPI

# SYNTAXE 3 : importer avec un alias (raccourci)
import numpy as np   # "numpy" est long à écrire, on le renomme "np"

# Utilisation avec l'alias
vecteur = np.array([1, 2, 3])   # np = raccourci pour numpy
```

## Dans notre projet

```python
# Dans app/embeddings.py — ligne par ligne expliquée

import chromadb
# → Charge la bibliothèque ChromaDB (base vectorielle)
# Sans ça, on ne peut pas créer de base vectorielle

from sentence_transformers import SentenceTransformer
# → Depuis la bibliothèque sentence_transformers,
#   on prend UNIQUEMENT la classe SentenceTransformer
# On n'a pas besoin du reste de la bibliothèque

from dotenv import load_dotenv
# → Pour lire les variables d'environnement depuis le fichier .env

import os
# → Module intégré Python (pas besoin de pip install)
# Permet de lire les variables d'environnement avec os.getenv()
```

---

# 2. Les variables et types de données

## C'est quoi ?

Une variable est une boîte avec un nom qui contient une valeur. Python a plusieurs types de boîtes selon ce qu'on met dedans.

## Comment ça marche ?

```python
# --- LES TYPES DE BASE ---

# str (string) : texte, toujours entre guillemets
nom = "Chatbot FAQ"
message = 'Bonjour !'           # guillemets simples ou doubles, même chose
long_texte = """Ceci est
un texte sur
plusieurs lignes"""              # triple guillemets pour les textes longs

# int (integer) : nombre entier
nombre_chunks = 42
index = 0

# float : nombre décimal
score = 0.95
temperature = 0.1

# bool (boolean) : vrai ou faux, UNIQUEMENT ces deux valeurs
est_vide = False
serveur_actif = True

# None : l'absence de valeur (comme NULL en SQL)
resultat = None       # "Je ne sais pas encore ce que ça vaut"

# --- VÉRIFIER LE TYPE ---
print(type(nom))          # <class 'str'>
print(type(nombre_chunks)) # <class 'int'>
print(type(0.95))         # <class 'float'>
print(type(True))         # <class 'bool'>
print(type(None))         # <class 'NoneType'>
```

## Dans notre projet

```python
# Dans app/chat.py

temperature = 0.1
# → float : contrôle la "créativité" du modèle
# 0.0 = toujours la même réponse (déterministe)
# 1.0 = réponses très variées et créatives

NOM_MODELE = "mistral"
# → str : le nom du modèle Ollama à utiliser
# En MAJUSCULES par convention = constante (valeur qui ne change pas)

historique = None
# → None : valeur par défaut quand pas d'historique fourni
# On verra comment gérer ça dans la section sur les fonctions
```

---

# 3. Les fonctions et def

## C'est quoi ?

Une fonction est un bloc de code réutilisable avec un nom. Au lieu de réécrire le même code 10 fois, tu l'écris une fois dans une fonction et tu l'appelles quand tu veux.

## Comment ça marche ?

```python
# DÉFINITION d'une fonction
# def = "define" (définir)
# nom_fonction = le nom qu'on donne
# (parametre1, parametre2) = les données qu'on lui donne à l'entrée
# -> str = le type de ce que la fonction retourne (optionnel mais utile)

def additionner(a, b):
    resultat = a + b
    return resultat    # return = "voici la réponse, j'ai fini"

# APPEL de la fonction
somme = additionner(3, 5)   # somme vaut maintenant 8
print(somme)                 # Affiche 8

# --- PARAMÈTRES PAR DÉFAUT ---
# On peut donner une valeur par défaut à un paramètre
# Si l'appelant ne précise pas ce paramètre, la valeur par défaut est utilisée

def saluer(nom, langue="fr"):
    if langue == "fr":
        return f"Bonjour {nom} !"
    else:
        return f"Hello {nom}!"

# Appels possibles :
print(saluer("Jean"))           # → "Bonjour Jean !"  (langue="fr" par défaut)
print(saluer("Jean", "en"))     # → "Hello Jean!"     (on a précisé la langue)
print(saluer("Jean", langue="en"))  # → même résultat, plus lisible

# --- FONCTIONS SANS RETOUR ---
# Si une fonction ne retourne rien, elle retourne None implicitement

def afficher_message(texte):
    print(f"📢 {texte}")
    # Pas de return → retourne None

afficher_message("Chargement...")   # Affiche "📢 Chargement..."

# --- FONCTIONS AVEC PLUSIEURS RETOURS ---
def diviser(a, b):
    if b == 0:
        return None, "Division par zéro impossible"
    return a / b, "OK"

resultat, message = diviser(10, 2)    # resultat=5.0, message="OK"
resultat, message = diviser(10, 0)    # resultat=None, message="Division par zéro impossible"
```

## Dans notre projet

```python
# Dans app/embeddings.py

def generer_embedding(texte: str) -> list[float]:
    #              ↑           ↑         ↑
    #         paramètre    son type   type retourné

    """
    Ceci est une docstring : documentation de la fonction.
    Elle explique ce que fait la fonction, ses paramètres et ce qu'elle retourne.
    Python l'affiche quand on fait help(generer_embedding)
    """

    vecteur = modele_embedding.encode(
        texte,
        normalize_embeddings=True
    )
    return vecteur.tolist()  # retourne la liste de nombres

# Appel de la fonction :
mon_vecteur = generer_embedding("Quels sont vos horaires ?")
# mon_vecteur est maintenant une liste de 1024 nombres


# Dans app/ingestion.py

def charger_document(chemin: str) -> list[str]:
    # Paramètre par défaut ? Non, ici chemin est obligatoire
    # Si on appelle charger_document() sans argument → erreur

    extension = Path(chemin).suffix.lower()

    if extension == ".pdf":
        texte = lire_pdf(chemin)       # Appelle une AUTRE fonction
    elif extension in [".txt", ".md"]:
        texte = lire_fichier_texte(chemin)  # Et une autre encore
    else:
        raise ValueError(f"Format non supporté")  # raise = déclencher une erreur

    chunks = decouper_en_chunks(texte)
    return chunks
```

---

# 4. Les f-strings

## C'est quoi ?

Un f-string (formatted string) permet d'insérer des variables directement dans du texte. Le **f** devant les guillemets active cette fonctionnalité.

## Comment ça marche ?

```python
nom = "Jean"
age = 25
ville = "Yaoundé"

# SANS f-string (vieux style, lourd à écrire)
message = "Bonjour " + nom + ", tu as " + str(age) + " ans."
# Note : str(age) obligatoire car on ne peut pas concaténer str et int

# AVEC f-string (moderne, lisible)
message = f"Bonjour {nom}, tu as {age} ans."
# Les accolades {} contiennent la variable à insérer
# Python convertit automatiquement en texte

# On peut mettre n'importe quelle expression dans {}
message = f"Dans 10 ans, tu auras {age + 10} ans."
message = f"Ville en majuscules : {ville.upper()}"
message = f"Pi vaut environ {3.14159:.2f}"  # :.2f = 2 décimales → "3.14"

# MULTI-LIGNE avec f-string
infos = f"""
Nom    : {nom}
Age    : {age}
Ville  : {ville}
"""

print(infos)
# Affiche :
# Nom    : Jean
# Age    : 25
# Ville  : Yaoundé
```

## Dans notre projet

```python
# Dans app/embeddings.py

ids.append(f"chunk_{index:04d}")
#                         ↑
#                   :04d = formater en entier sur 4 chiffres avec des zéros devant
# index=0   → "chunk_0000"
# index=1   → "chunk_0001"
# index=42  → "chunk_0042"
# Pourquoi ? Pour que les IDs soient triés correctement (0001 avant 0010)


print(f"✅ {len(chunks)} chunks indexés dans ChromaDB !")
# len(chunks) = nombre d'éléments dans la liste chunks
# Si chunks a 15 éléments → "✅ 15 chunks indexés dans ChromaDB !"


# Dans app/chat.py — Construction du prompt

messages.append({
    "role": "user",
    "content": f"""Voici les informations disponibles :

<contexte>
{contexte}
</contexte>

Ma question : {question}"""
})
# Le triple guillemet """ permet un texte sur plusieurs lignes
# {contexte} = le texte des chunks retrouvés dans ChromaDB
# {question} = la question posée par l'utilisateur
# Tout ça est inséré dynamiquement dans le prompt envoyé à Mistral
```

---

# 5. Les listes et leurs opérations

## C'est quoi ?

Une liste est une collection ordonnée d'éléments. Elle peut contenir n'importe quel type de données, même mélangés.

## Comment ça marche ?

```python
# CRÉER une liste
fruits = ["mangue", "avocat", "banane"]
nombres = [1, 2, 3, 4, 5]
mixte = ["texte", 42, True, None]   # On peut mélanger les types
vide = []                            # Liste vide

# ACCÉDER aux éléments (index commence à 0 !)
print(fruits[0])    # "mangue"   (premier élément)
print(fruits[1])    # "avocat"   (deuxième)
print(fruits[-1])   # "banane"   (dernier, index négatif)
print(fruits[-2])   # "avocat"   (avant-dernier)

# MODIFIER
fruits[0] = "papaye"          # Remplace "mangue" par "papaye"

# AJOUTER
fruits.append("orange")       # Ajoute à la FIN → ["papaye", "avocat", "banane", "orange"]
fruits.insert(0, "citron")    # Insère au début (index 0)

# SUPPRIMER
fruits.remove("avocat")       # Supprime par valeur
del fruits[0]                 # Supprime par index
dernier = fruits.pop()        # Supprime et RETOURNE le dernier élément

# LONGUEUR
print(len(fruits))            # Nombre d'éléments

# PARCOURIR (boucle for)
for fruit in fruits:
    print(f"J'aime les {fruit}s")

# PARCOURIR avec index
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# enumerate() donne en même temps l'index ET la valeur

# SLICING (extraire une sous-liste)
nombres = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(nombres[2:5])     # [2, 3, 4]   (de l'index 2 inclus à 5 exclu)
print(nombres[:3])      # [0, 1, 2]   (du début à l'index 3 exclu)
print(nombres[7:])      # [7, 8, 9]   (de l'index 7 à la fin)

# VÉRIFIER si un élément est dans la liste
if "mangue" in fruits:
    print("Il y a des mangues !")

# JOINDRE une liste de strings en une seule string
mots = ["Bonjour", "tout", "le", "monde"]
phrase = " ".join(mots)     # "Bonjour tout le monde"
textes = ["Partie 1", "Partie 2", "Partie 3"]
joint = "\n\n---\n\n".join(textes)   # Joints avec séparateur
```

## Dans notre projet

```python
# Dans app/embeddings.py

# On crée des listes vides pour les remplir ensuite
documents  = []
embeddings = []
ids        = []
metadatas  = []

# On remplit les 4 listes en parallèle avec une boucle
for index, chunk in enumerate(chunks):
    # enumerate(chunks) donne (0, chunk0), (1, chunk1), etc.

    vecteur = generer_embedding(chunk)   # Calcule le vecteur pour ce chunk

    documents.append(chunk)              # Ajoute le texte
    embeddings.append(vecteur)           # Ajoute le vecteur correspondant
    ids.append(f"chunk_{index:04d}")     # Ajoute l'ID unique
    metadatas.append({                   # Ajoute les métadonnées
        "source": nom_source,
        "index": index
    })

# Résultat : 4 listes de même taille, alignées par index
# documents[0]  ↔  embeddings[0]  ↔  ids[0]  ↔  metadatas[0]
# documents[1]  ↔  embeddings[1]  ↔  ids[1]  ↔  metadatas[1]
# etc.

# Dans app/chat.py

contexte = "\n\n---\n\n".join(chunks_pertinents)
# chunks_pertinents = ["texte chunk 1", "texte chunk 2", "texte chunk 3"]
# join() colle tout avec le séparateur "\n\n---\n\n" entre chaque
# Résultat :
# "texte chunk 1
#
# ---
#
# texte chunk 2
#
# ---
#
# texte chunk 3"
```

---

# 6. Les dictionnaires

## C'est quoi ?

Un dictionnaire stocke des paires **clé: valeur**. Comme un vrai dictionnaire : tu cherches un mot (clé) et tu obtiens sa définition (valeur).

## Comment ça marche ?

```python
# CRÉER un dictionnaire
personne = {
    "nom": "Jean",        # clé: "nom"    → valeur: "Jean"
    "age": 25,            # clé: "age"    → valeur: 25
    "ville": "Yaoundé",   # clé: "ville"  → valeur: "Yaoundé"
    "actif": True         # clé: "actif"  → valeur: True
}

# ACCÉDER à une valeur
print(personne["nom"])      # "Jean"
print(personne["age"])      # 25

# ACCÉDER avec valeur par défaut (si la clé n'existe pas)
print(personne.get("email", "pas d'email"))  # "pas d'email" (email n'existe pas)

# MODIFIER / AJOUTER
personne["age"] = 26              # Modifie une valeur existante
personne["email"] = "j@j.cm"     # Ajoute une nouvelle paire

# SUPPRIMER
del personne["actif"]             # Supprime la paire "actif"

# VÉRIFIER si une clé existe
if "nom" in personne:
    print("La clé 'nom' existe")

# PARCOURIR
for cle, valeur in personne.items():
    print(f"{cle} = {valeur}")

# Lister les clés
print(personne.keys())    # dict_keys(['nom', 'age', 'ville', 'email'])

# Lister les valeurs
print(personne.values())  # dict_values(['Jean', 26, 'Yaoundé', 'j@j.cm'])

# --- DICTIONNAIRE IMBRIQUÉ (dict dans un dict) ---
utilisateur = {
    "profil": {
        "nom": "Jean",
        "age": 25
    },
    "parametres": {
        "langue": "fr",
        "notifications": True
    }
}

# Accès imbriqué
print(utilisateur["profil"]["nom"])              # "Jean"
print(utilisateur["parametres"]["langue"])        # "fr"

# --- LISTE DE DICTIONNAIRES (très courant en API) ---
messages = [
    {"role": "system",    "content": "Tu es un assistant."},
    {"role": "user",      "content": "Bonjour !"},
    {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider ?"}
]

# Accès
print(messages[0]["role"])      # "system"
print(messages[1]["content"])   # "Bonjour !"
```

## Dans notre projet

```python
# Dans app/chat.py — Les messages pour l'API Mistral

# C'est une LISTE DE DICTIONNAIRES
messages = [
    {
        "role": "system",         # clé "role" → valeur "system"
        "content": "Tu es..."     # clé "content" → valeur le prompt
    }
]

# On ajoute des éléments à la liste avec append()
messages.append({
    "role": "user",
    "content": f"<contexte>{contexte}</contexte>\n\nMa question : {question}"
})
# → messages est maintenant une liste de 2 dictionnaires

# Dans app/embeddings.py — Les métadonnées

metadatas.append({
    "source": nom_source,   # ex: "faq.txt"
    "index": index          # ex: 0, 1, 2, ...
})

# Dans app/chat.py — La réponse retournée par la fonction

return {
    "reponse": texte_reponse,      # La réponse de Mistral
    "sources": chunks_pertinents,  # Les chunks utilisés
    "modele": NOM_MODELE,          # "mistral"
    "tokens_utilises": 0
}
# On retourne un dictionnaire → l'appelant peut accéder à chaque partie :
# resultat["reponse"]  → le texte de la réponse
# resultat["sources"]  → la liste des chunks
```

---

# 7. Le typage (annotations de type)

## C'est quoi ?

Python est un langage à typage dynamique : tu n'es pas obligé de préciser le type d'une variable. Mais depuis Python 3.5+, on peut ajouter des **annotations de type** pour rendre le code plus lisible et détecter les erreurs.

## Comment ça marche ?

```python
# SANS annotation (Python basique)
def generer_embedding(texte):
    return [0.1, 0.2, 0.3]

# AVEC annotation (Python moderne)
def generer_embedding(texte: str) -> list[float]:
    #                         ↑           ↑
    #               type du paramètre   type du retour
    return [0.1, 0.2, 0.3]

# Les annotations les plus courantes :
def exemples(
    a: str,             # string
    b: int,             # entier
    c: float,           # décimal
    d: bool,            # booléen
    e: list,            # liste (quelconque)
    f: list[str],       # liste de strings
    g: list[float],     # liste de floats
    h: dict,            # dictionnaire (quelconque)
    i: dict[str, int],  # dict avec clés str et valeurs int
) -> str:               # cette fonction retourne une string
    pass

# TYPES OPTIONNELS avec None
# Si un paramètre peut être None OU un certain type :
from typing import Optional

def fonction(historique: list[dict] = None):
    # historique peut être None ou une liste de dicts
    pass

# En Python 3.10+ on peut écrire :
def fonction(historique: list[dict] | None = None):
    # | None signifie "ou None"
    pass

# IMPORTANT : Les annotations ne sont PAS vérifiées par Python à l'exécution !
# Ce sont des INDICATIONS pour le développeur et les outils (VS Code, etc.)
def mauvaise_fonction(x: int) -> str:
    return x + 1    # Retourne un int, pas une str → Python ne plante pas
                    # Mais VS Code te le signale en rouge
```

## Dans notre projet

```python
# Dans app/embeddings.py

def generer_embedding(texte: str) -> list[float]:
#                            ↑           ↑
#                    attend une str   retourne une liste de floats
    vecteur = modele_embedding.encode(texte, normalize_embeddings=True)
    return vecteur.tolist()


def indexer_documents(chunks: list[str], nom_source: str = "document"):
#                            ↑                      ↑
#               liste de strings      string avec valeur par défaut
#  → pas de -> car la fonction ne retourne rien (retourne None implicitement)


def rechercher_chunks_pertinents(question: str, n_resultats: int = 3) -> list[str]:
#                                         ↑               ↑              ↑
#                                       string         entier=3    liste de strings


# Dans app/ingestion.py

def decouper_en_chunks(texte: str, taille: int = 500, chevauchement: int = 50) -> list[str]:
#                             ↑             ↑                        ↑             ↑
#                           string       entier=500              entier=50    liste de strings
```

---

# 8. Les classes et Pydantic

## C'est quoi ?

Une classe est un modèle pour créer des objets. Elle regroupe des données (attributs) et des fonctions (méthodes) qui vont ensemble.

## Comment ça marche ?

```python
# CLASSE SIMPLE
class Voiture:
    # __init__ = constructeur : appelé quand on crée une voiture
    def __init__(self, marque, couleur, annee):
        self.marque = marque      # self = "cet objet lui-même"
        self.couleur = couleur    # self.marque = attribut de l'objet
        self.annee = annee

    # Méthode = fonction appartenant à la classe
    def description(self):
        return f"{self.annee} - {self.marque} ({self.couleur})"

    def age(self, annee_actuelle):
        return annee_actuelle - self.annee

# CRÉER des instances (objets)
ma_voiture = Voiture("Toyota", "rouge", 2020)
autre_voiture = Voiture("Honda", "blanche", 2018)

# UTILISER les attributs et méthodes
print(ma_voiture.marque)           # "Toyota"
print(ma_voiture.description())    # "2020 - Toyota (rouge)"
print(ma_voiture.age(2025))        # 5


# --- HÉRITAGE ---
# Une classe peut hériter d'une autre et en étendre les fonctionnalités

class VoitureElectrique(Voiture):   # hérite de Voiture
    def __init__(self, marque, couleur, annee, autonomie_km):
        super().__init__(marque, couleur, annee)  # appelle le __init__ du parent
        self.autonomie_km = autonomie_km          # attribut supplémentaire

    def autonomie_info(self):
        return f"Autonomie : {self.autonomie_km} km"

tesla = VoitureElectrique("Tesla", "noir", 2023, 500)
print(tesla.description())      # Héritée de Voiture : "2023 - Tesla (noir)"
print(tesla.autonomie_info())   # Propre à VoitureElectrique
```

## Pydantic : des classes spéciales pour valider les données

```python
# Pydantic est une bibliothèque qui crée des classes avec VALIDATION automatique
from pydantic import BaseModel, Field

# On hérite de BaseModel (classe de Pydantic)
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,              # "..." = champ obligatoire (pas de valeur par défaut)
        min_length=3,     # La question doit avoir au moins 3 caractères
        max_length=500,   # Et au plus 500 caractères
        description="La question à poser au chatbot"
    )
    historique: list[dict] = Field(
        default=[],       # Valeur par défaut : liste vide
        description="Historique optionnel"
    )

# UTILISATION
# Si on envoie des données valides :
req = QuestionRequest(question="Quels sont vos horaires ?")
print(req.question)      # "Quels sont vos horaires ?"
print(req.historique)    # []  (valeur par défaut)

# Si on envoie des données INVALIDES :
req = QuestionRequest(question="Hi")  # 2 caractères < min_length=3 → ERREUR automatique
# Pydantic lève une ValidationError avec un message clair

# C'est FastAPI qui utilise Pydantic automatiquement :
# Quand un client envoie {"question": "..."} à ton API,
# FastAPI crée un QuestionRequest et valide les données AVANT que ton code s'exécute
```

## Dans notre projet

```python
# Dans app/main.py

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,           # Obligatoire
        min_length=3,
        max_length=500,
    )
    historique: list[dict] = Field(default=[])
    # Si le client n'envoie pas "historique", ça vaut [] automatiquement

class ReponseChatbot(BaseModel):
    reponse: str         # La réponse textuelle
    sources: list[str]   # Les chunks utilisés
    tokens_utilises: int # Nombre de tokens

# FastAPI utilise ces classes pour :
# 1. VALIDER les données entrantes (QuestionRequest)
# 2. FORMATER les données sortantes (ReponseChatbot)
# 3. DOCUMENTER l'API automatiquement dans /docs
```

---

# 9. Les décorateurs (@)

## C'est quoi ?

Un décorateur modifie le comportement d'une fonction sans changer son code. Le symbole `@` devant une fonction indique un décorateur.

Analogie : imagine que tu as une fonction qui fait du café. Un décorateur pourrait automatiquement chauffer la tasse AVANT et nettoyer la machine APRÈS, sans que la fonction "faire le café" sache que ça se passe.

## Comment ça marche ?

```python
# DÉCORATEUR SIMPLE — comprendre le principe

def mon_decorateur(fonction_originale):
    def fonction_modifiee():
        print("Avant l'exécution")
        resultat = fonction_originale()   # Appelle la vraie fonction
        print("Après l'exécution")
        return resultat
    return fonction_modifiee

# SANS décorateur :
def dire_bonjour():
    return "Bonjour !"

# AVEC décorateur (syntaxe @) :
@mon_decorateur
def dire_bonjour():
    return "Bonjour !"

# Ces deux écritures sont ÉQUIVALENTES :
# @mon_decorateur          ≡     dire_bonjour = mon_decorateur(dire_bonjour)
# def dire_bonjour(): ...

# Appel :
resultat = dire_bonjour()
# Affiche : "Avant l'exécution"
# Affiche : "Après l'exécution"
# resultat = "Bonjour !"
```

## Dans notre projet : les décorateurs FastAPI

```python
from fastapi import FastAPI
app = FastAPI()

# @app.get("/")
# Signifie : "Quand quelqu'un fait une requête GET à l'URL '/', exécute cette fonction"
@app.get("/")
async def racine():
    return {"message": "API opérationnelle"}

# @app.post("/chat")
# Signifie : "Quand quelqu'un fait une requête POST à '/chat', exécute cette fonction"
@app.post("/chat", response_model=ReponseChatbot)
async def chat(request: QuestionRequest):
    # response_model=ReponseChatbot = "formate la réponse selon ce modèle Pydantic"
    resultat = repondre_a_question(
        question=request.question,
        historique=request.historique
    )
    return ReponseChatbot(**resultat)

# @app.post("/document/upload")
# Signifie : requête POST à '/document/upload'
@app.post("/document/upload")
async def uploader_document(fichier: UploadFile = File(...)):
    # UploadFile = type spécial FastAPI pour les fichiers uploadés
    # File(...) = "ce paramètre vient du formulaire d'upload" (obligatoire)
    pass

# RÉSUMÉ des décorateurs FastAPI :
# @app.get(url)    → répond aux requêtes GET    (récupérer des données)
# @app.post(url)   → répond aux requêtes POST   (envoyer des données)
# @app.put(url)    → répond aux requêtes PUT    (modifier)
# @app.delete(url) → répond aux requêtes DELETE (supprimer)
```

---

# 10. async / await

## C'est quoi ?

Par défaut, Python exécute le code ligne par ligne, en attendant que chaque opération se termine avant de passer à la suivante. C'est du code **synchrone**.

Le problème : si 100 utilisateurs posent une question en même temps, le premier attendrait que les 99 autres soient traités. Une catastrophe pour une API.

`async/await` permet de faire du code **asynchrone** : pendant qu'on attend la réponse de Mistral (qui prend 2-3 secondes), Python peut traiter d'autres requêtes.

## Comment ça marche ?

```python
import asyncio

# Fonction SYNCHRONE (bloquante)
def telecharger_fichier():
    # Pendant le téléchargement, Python attend et ne fait rien d'autre
    time.sleep(3)   # Simule 3 secondes de téléchargement
    return "fichier téléchargé"

# Fonction ASYNCHRONE (non-bloquante)
async def telecharger_fichier_async():
    # async def = "cette fonction peut être mise en pause"
    await asyncio.sleep(3)   # await = "attends, mais laisse les autres s'exécuter"
    return "fichier téléchargé"

# RÈGLE D'OR :
# Une fonction async ne peut être appelée qu'avec await
# await ne peut être utilisé que dans une fonction async

async def main():
    resultat = await telecharger_fichier_async()
    print(resultat)

asyncio.run(main())   # Lance la boucle asynchrone


# --- ANALOGIE ---
# Synchrone = un seul caissier qui sert les clients un par un
# Le client 2 attend que le client 1 soit servi
#
# Asynchrone = un caissier qui dit au client 1 "votre commande est en préparation,
# revenez dans 3 minutes" et sert le client 2 pendant ce temps
```

## Dans notre projet

```python
# Dans app/main.py

# FastAPI utilise async par défaut pour toutes ses fonctions
# Ça permet de gérer plusieurs requêtes simultanément

@app.get("/")
async def racine():                    # async def = fonction asynchrone
    return {"message": "OK"}

@app.post("/chat")
async def chat(request: QuestionRequest):
    # Pendant qu'on attend la réponse de Mistral (via Ollama),
    # FastAPI peut traiter d'autres requêtes entrantes
    resultat = repondre_a_question(    # Appel SANS await car repondre_a_question
        question=request.question,     # n'est PAS async (elle utilise le SDK OpenAI
        historique=request.historique  # qui est synchrone)
    )
    return ReponseChatbot(**resultat)

# NOTE : Dans notre projet, les appels à Ollama ne sont pas vraiment async
# (le SDK OpenAI utilisé est synchrone).
# Pour un vrai projet en production, on utiliserait httpx.AsyncClient.
# Pour apprendre, c'est largement suffisant comme ça.
```

---

# 11. Les gestionnaires de contexte (with)

## C'est quoi ?

`with` est un mot-clé Python qui gère automatiquement l'ouverture et la fermeture de ressources (fichiers, connexions réseau, bases de données).

## Comment ça marche ?

```python
# SANS with (à l'ancienne, risqué)
fichier = open("faq.txt", "r")
contenu = fichier.read()
fichier.close()    # Si une erreur survient AVANT cette ligne, le fichier reste ouvert !

# AVEC with (moderne, sûr)
with open("faq.txt", "r", encoding="utf-8") as f:
    contenu = f.read()
    # Le fichier est automatiquement fermé quand on sort du bloc with
    # MÊME si une erreur survient

# Explication de la syntaxe :
# with [expression] as [variable]:
#     [code qui utilise la variable]
# → quand on sort du bloc (normalement ou sur erreur), Python appelle .close() automatiquement


# Autres usages courants de with :
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        texte = page.extract_text()
# Le PDF est fermé automatiquement après la boucle


# Pour les connexions réseau ou base de données (même principe)
import sqlite3

with sqlite3.connect("ma_base.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clients")
    resultats = cursor.fetchall()
# La connexion est fermée automatiquement
```

## Dans notre projet

```python
# Dans app/ingestion.py

# Lecture d'un fichier texte
with open(chemin_fichier, "r", encoding="utf-8") as f:
    return f.read()
# "r" = mode lecture (read)
# encoding="utf-8" = important pour les accents et caractères spéciaux
# f = le fichier ouvert (convention de nommer le fichier "f")
# Après le with : le fichier est fermé automatiquement


# Lecture d'un PDF
with pdfplumber.open(chemin) as pdf:
    for numero_page, page in enumerate(pdf.pages):
        texte = page.extract_text()
        if texte:
            texte_complet.append(texte)
# Le PDF est fermé automatiquement, même si une erreur survient

# Lors de l'upload d'un fichier via FastAPI
with open(chemin_fichier, "wb") as buffer:
    shutil.copyfileobj(fichier.file, buffer)
# "wb" = write binary (écriture en mode binaire, pour les fichiers non-texte)
# buffer = le fichier de destination
# shutil.copyfileobj copie les données du fichier source vers le buffer
```

---

# 12. La gestion des erreurs (try/except)

## C'est quoi ?

Quand une erreur survient en Python, le programme s'arrête et affiche un message d'erreur (traceback). `try/except` permet d'intercepter ces erreurs et de les gérer proprement.

## Comment ça marche ?

```python
# SANS gestion d'erreur
nombre = int("abc")   # ValueError : impossible de convertir "abc" en entier
print("Ce code ne s'exécute jamais")   # Le programme s'arrête à la ligne du dessus

# AVEC try/except
try:
    nombre = int("abc")     # Python essaie d'exécuter ce code
    print("Conversion OK")
except ValueError:           # Si une ValueError survient...
    print("Impossible de convertir en nombre")  # ...on fait ça à la place
    nombre = 0               # Valeur par défaut

print("Le programme continue")  # Ça s'exécute maintenant


# PLUSIEURS types d'erreurs
try:
    fichier = open("inexistant.txt", "r")
    contenu = fichier.read()
    nombre = int(contenu)
except FileNotFoundError:    # Si le fichier n'existe pas
    print("Fichier introuvable")
except ValueError:           # Si le contenu n'est pas un nombre
    print("Le fichier ne contient pas un nombre")
except Exception as e:       # N'importe quelle autre erreur
    print(f"Erreur inattendue : {e}")   # e = l'objet erreur avec le message


# FINALLY : s'exécute toujours, erreur ou pas
try:
    conn = ouvrir_connexion()
    resultat = conn.query("SELECT ...")
except Exception as e:
    print(f"Erreur : {e}")
finally:
    conn.close()   # Toujours fermer la connexion, même si ça a planté


# RAISE : déclencher une erreur volontairement
def diviser(a, b):
    if b == 0:
        raise ValueError("Impossible de diviser par zéro")
    return a / b
```

## Dans notre projet

```python
# Dans app/main.py

@app.post("/chat")
async def chat(request: QuestionRequest):
    try:
        # On essaie de répondre à la question
        resultat = repondre_a_question(
            question=request.question,
            historique=request.historique
        )
        return ReponseChatbot(**resultat)

    except Exception as e:
        # Si QUOI QUE CE SOIT plante (Ollama éteint, ChromaDB vide, etc.)
        # On retourne une erreur HTTP 500 avec un message clair
        # au lieu de laisser l'API crasher
        raise HTTPException(
            status_code=500,        # Code HTTP "Erreur serveur"
            detail=f"Erreur lors du traitement : {str(e)}"
            # str(e) = convertit l'erreur en texte lisible
        )

# Dans app/ingestion.py

def charger_document(chemin: str) -> list[str]:
    chemin_fichier = Path(chemin)

    if not chemin_fichier.exists():
        # raise = déclenche volontairement une erreur
        # FileNotFoundError = type d'erreur standard Python pour "fichier introuvable"
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    extension = Path(chemin).suffix.lower()
    if extension not in [".pdf", ".txt", ".md"]:
        # ValueError = type d'erreur pour "valeur invalide"
        raise ValueError(f"Format non supporté : {extension}")
```

---

# 13. Les variables d'environnement et dotenv

## C'est quoi ?

Des valeurs de configuration stockées EN DEHORS du code. Ça permet de changer la configuration sans modifier le code, et de ne pas exposer des infos sensibles sur GitHub.

## Comment ça marche ?

```python
# FICHIER .env (dans le dossier du projet)
# Ce fichier contient les variables sous forme CLE=VALEUR
# Ne JAMAIS committer ce fichier sur GitHub !

# Contenu de .env :
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_MODEL=mistral
# NOM_APP=Mon Chatbot


# DANS LE CODE PYTHON

from dotenv import load_dotenv  # Bibliothèque pour lire .env
import os                        # Module standard pour accéder aux variables d'env

# 1. Charger le fichier .env (rend les variables disponibles)
load_dotenv()
# Sans cette ligne, os.getenv() ne trouvera pas les variables du fichier .env

# 2. Lire une variable
url = os.getenv("OLLAMA_BASE_URL")
# → "http://localhost:11434/v1"

# 3. Lire avec valeur par défaut (si la variable n'existe pas)
modele = os.getenv("OLLAMA_MODEL", "mistral")
# → Si OLLAMA_MODEL existe dans .env → retourne sa valeur
# → Sinon → retourne "mistral" (valeur par défaut)

# 4. Vérifier qu'une variable obligatoire est définie
db_url = os.getenv("DATABASE_URL")
if db_url is None:
    raise ValueError("La variable DATABASE_URL est manquante dans .env")
```

## Dans notre projet

```python
# Dans app/embeddings.py et app/chat.py

from dotenv import load_dotenv
import os

load_dotenv()   # Charge les variables de .env dans l'environnement

# Lecture des variables de configuration
NOM_MODELE = os.getenv("OLLAMA_MODEL", "mistral")
# Si .env contient OLLAMA_MODEL=mistral-small3 → NOM_MODELE = "mistral-small3"
# Si .env ne définit pas OLLAMA_MODEL          → NOM_MODELE = "mistral" (défaut)

# Dans app/chat.py

from openai import OpenAI

client_ollama = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    # os.getenv récupère l'URL depuis .env
    # Si non définie → utilise "http://localhost:11434/v1" par défaut
    api_key="ollama"
)
```

---

# 14. La structure en modules

## C'est quoi ?

Dans notre projet, le code est réparti sur plusieurs fichiers (modules). Les imports avec des points (`.`) permettent à un fichier d'utiliser le code d'un autre fichier du même projet.

## Comment ça marche ?

```
chatbot-faq/
└── app/
    ├── __init__.py     ← rend le dossier "app" importable
    ├── main.py
    ├── ingestion.py
    ├── embeddings.py
    └── chat.py
```

```python
# __init__.py
# Ce fichier (souvent vide) dit à Python :
# "Le dossier 'app' est un package (ensemble de modules)"
# Sans lui, les imports avec des points ne fonctionnent pas

# --- LES IMPORTS RELATIFS (avec des points) ---

# Dans app/chat.py, pour utiliser une fonction de app/embeddings.py :
from .embeddings import rechercher_chunks_pertinents
#    ↑
#    Le point "." signifie "dans le même dossier que moi"
#    Donc : depuis le fichier embeddings.py dans le même dossier (app/)
#           importer la fonction rechercher_chunks_pertinents

# Dans app/main.py :
from .ingestion import charger_document
from .embeddings import indexer_documents
from .chat import repondre_a_question
# Chaque "." pointe vers le dossier "app/"


# --- IMPORTS ABSOLUS (sans point) ---
# Utilisés pour les bibliothèques installées (chromadb, fastapi, etc.)
import chromadb
from fastapi import FastAPI


# --- POURQUOI SÉPARER EN MODULES ? ---
# 1. Lisibilité : chaque fichier a une responsabilité claire
# 2. Maintenance : modifier les embeddings ne touche pas au chat
# 3. Réutilisabilité : d'autres fichiers peuvent importer les mêmes fonctions
# 4. Collaboration : plusieurs développeurs travaillent sur des fichiers différents
```

---

# 15. List Comprehensions

## C'est quoi ?

Une syntaxe Python compacte pour créer des listes en une seule ligne, au lieu d'écrire une boucle for complète.

## Comment ça marche ?

```python
# VERSION LONGUE (boucle for classique)
nombres = [1, 2, 3, 4, 5]
carres = []
for n in nombres:
    carres.append(n ** 2)
# carres = [1, 4, 9, 16, 25]

# VERSION COMPACTE (list comprehension)
carres = [n ** 2 for n in nombres]
# Lire : "pour chaque n dans nombres, calcule n²"
# carres = [1, 4, 9, 16, 25]


# AVEC CONDITION (filtre)
pairs = [n for n in nombres if n % 2 == 0]
# Lire : "pour chaque n dans nombres, SI n est pair"
# pairs = [2, 4]


# EXEMPLES CONCRETS
textes = ["  Bonjour  ", "  monde  ", "  !  "]

# Nettoyer les espaces de chaque texte
propres = [t.strip() for t in textes]
# propres = ["Bonjour", "monde", "!"]

# Mettre en majuscules
majuscules = [t.strip().upper() for t in textes]
# majuscules = ["BONJOUR", "MONDE", "!"]
```

## Dans notre projet

```python
# Dans app/embeddings.py

# Créer les IDs pour tous les chunks d'un coup
ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
# range(len(chunks)) = [0, 1, 2, 3, ...]
# Pour chaque i → f"chunk_{i:04d}"
# Résultat : ["chunk_0000", "chunk_0001", "chunk_0002", ...]

# Créer les métadonnées pour tous les chunks
metadatas = [{"source": nom_source, "index": i} for i in range(len(chunks))]
# Pour chaque i → {"source": "faq.txt", "index": i}
# Résultat : [{"source": "faq.txt", "index": 0}, {"source": "faq.txt", "index": 1}, ...]

# Convertir les vecteurs numpy en listes Python
return [v.tolist() for v in vecteurs]
# Pour chaque vecteur numpy v → v.tolist() (convertit en liste Python)


# Dans app/chat.py

# Construire l'historique pour l'API à partir des messages stockés
historique_api = [
    {"role": msg["role"], "content": msg["contenu"]}
    for msg in historique
]
# Pour chaque message dans l'historique
# → crée un dict avec "role" et "content"
```

---

# RÉCAPITULATIF — Ce que fait chaque fichier du projet

```
app/
├── __init__.py
│   └── Vide. Rend "app" importable avec des points.
│
├── ingestion.py        "Je lis et découpe les documents"
│   ├── lire_fichier_texte(chemin)    → str
│   ├── lire_pdf(chemin)              → str
│   ├── decouper_en_chunks(texte)     → list[str]
│   └── charger_document(chemin)      → list[str]  ← fonction principale
│
├── embeddings.py       "Je transforme le texte en vecteurs et je stocke dans ChromaDB"
│   ├── [variable globale] modele_embedding = SentenceTransformer(...)
│   ├── generer_embedding(texte)              → list[float]
│   ├── generer_embeddings_batch(textes)      → list[list[float]]
│   ├── indexer_documents(chunks, nom_source) → None
│   └── rechercher_chunks_pertinents(question) → list[str]
│
├── chat.py             "Je construis le prompt et appelle Mistral"
│   ├── [variable globale] client_ollama = OpenAI(...)
│   ├── construire_prompt_systeme()           → str
│   └── repondre_a_question(question, historique) → dict
│
└── main.py             "Je gère les requêtes HTTP entrantes"
    ├── [classes Pydantic] QuestionRequest, ReponseChatbot
    ├── GET  /           → "API opérationnelle"
    ├── POST /document/upload  → indexe un document
    └── POST /chat             → retourne une réponse du chatbot
```

---

*Avec ces 15 concepts, tu comprends maintenant 95% de la syntaxe utilisée dans le Projet 1. Le reste (numpy arrays, shutil, Path) est intuitif une fois qu'on maîtrise les bases.*
