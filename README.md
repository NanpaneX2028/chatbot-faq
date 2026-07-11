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