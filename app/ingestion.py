import pdfplumber
from pathlib import Path

def lire_fichier_texte(chemin: str) -> str:
    """
    Lit un fichier .txt et retourne son contenu.

    :param chemin: Le chemin du fichier
    :return:
    Le contenu du fichier .txt sous forme de chaine de caractères
    """
    # Path() permet de manipuler les chemins de façons portable
    chemin_fichier = Path(chemin)

    # Vérification si le fichier existe
    if not chemin_fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable :{chemin}")

    # encoding="utf-8" est important pour les caractères spéciaux
    with open(chemin_fichier, "r",
              encoding="utf-8",) as f:
        return f.read()

def lire_pdf(chemin: str) -> str:
    """
    Extrait tout le texte d'un fichier PDF.
    :param chemin: Le chemin vers le fichier PDF
    :return:
    Tout le texte du PDF concatené
    """
    texte_complet = []

    # pdfplumber ouvre le PDF et permet d'accéder page par page
    with pdfplumber.open(chemin) as pdf:
        for numero_page, page in enumerate(pdf.pages):

            # extract_text() retourne le texte de la page ou None si elle est vide
            texte = page.extract_text()
            if texte: # On ignore les pages sans texte
                texte_complet.append(texte)

    # "\n\n" entre les pages pour marquer les séparations
    return "\n\n".join(texte_complet)

def decouper_en_chunks(texte: str, taille: int =500, chevauchement: int = 50) -> list[str]:

    """
    Découpe un long texte en morceaux de taille fixe avec chevauchement.

    Pourquoi le chevauchement ?
    Si on découpe sans chevauchement, une information importante peut se retrouver
    coupée entre deux chunks. Le chevauchement assure la continuité du contexte.

    Exemple avec taille = 20, chevauchement = 5:
    Texte : "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    Chunk 1 : "ABCDEFGHIJKLMNOPQRST"
    Chunk 2 : "PQRSTUVWXYZ..."

    :param texte: Le texte source à découper
    :param taille: Nombre de caractères par chunk(500 est un bon compromis)
    :param chevauchement: Nombre de caractères partagées entre chunk consécutifs
    :return: Liste de chunks (morceaux de texte)
    """
    chunks = []

    # On commence au début du texte
    debut = 0

    while debut< len(texte):
        # La fin du chunk est soit début+taille, soit la fin du texte
        fin = min(debut+taille, len(texte))

        chunk = texte[debut:fin].strip()

        if len(chunk) >= 20:
            chunks.append(chunk)

        debut += taille - chevauchement

    return chunks

def charger_document(chemin: str) -> list[str]:

    """
    Fonction principale : charge un document et retourne la liste de chunks
    Détecte automatiquement si c'est un PDF ou un fichier texte

    :param chemin: Chemin vers le fichier (.txt ou .pdf)

    :return: Liste de chunks prêts à être indexés
    """

    extension = Path(chemin).suffix.lower()

    if extension == ".pdf":
        texte = lire_pdf(chemin)
    elif extension in [".txt",".md"]:
        texte = lire_fichier_texte(chemin)
    else:
        raise ValueError(f"Format non supporté : {extension}. Utilisez .pdf ou .txt ou .md")

    chunks = decouper_en_chunks(texte)

    print(f"✅ Document chargé : {len(chunks)} chunks créés depuis '{chemin}'")
    return chunks