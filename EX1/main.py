import os
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def charger_instance(chemin_fichier):
    try:
        with open(chemin_fichier, 'r') as fichier:
            contenu = fichier.readlines()
            matrice_poids = []
            nombre_elements = None
            liste_largeurs = None

            for idx, ligne in enumerate(contenu):
                valeurs = list(map(int, ligne.strip().split()))
                if idx == 0:
                    nombre_elements = valeurs[0]
                elif idx == 1:
                    liste_largeurs = valeurs[:]
                else:
                    matrice_poids.append(valeurs)

            print("Matrice lue initialement :")
            for ligne in matrice_poids:
                print(ligne)

            for i in range(len(matrice_poids)):
                for j in range(len(matrice_poids)):
                    if matrice_poids[i][j] == 0 and matrice_poids[j][i] != 0:
                        matrice_poids[i][j] = matrice_poids[j][i]

            print("\nMatrice après correction de symétrie :")
            for ligne in matrice_poids:
                print(ligne)

            return nombre_elements, liste_largeurs, matrice_poids
    except FileNotFoundError:
        print(f"Impossible de trouver le fichier : {chemin_fichier}")
        print(f"Répertoire courant : {os.getcwd()}")
        raise


def evaluer_solution(ordre, largeurs, poids):
    positions = [sum(largeurs[:k]) + largeurs[k]/2 for k in ordre]
    score = 0
    for a in range(len(ordre)):
        for b in range(a + 1, len(ordre)):
            diff = abs(positions[a] - positions[b])
            score += poids[ordre[a]][ordre[b]] * diff
    return score


def traiter_segment(liste_permutations, largeurs, poids):
    meilleur_score = float('inf')
    meilleure_perm = None
    for candidate in liste_permutations:
        cout = evaluer_solution(candidate, largeurs, poids)
        if cout < meilleur_score:
            meilleur_score = cout
            meilleure_perm = candidate
    return meilleur_score, meilleure_perm


def recherche_parallele(n, largeurs, poids):
    toutes_permutations = list(permutations(range(n)))
    taille_bloc = max(1, len(toutes_permutations) // os.cpu_count())
    sous_ensembles = [toutes_permutations[i:i+taille_bloc] for i in range(0, len(toutes_permutations), taille_bloc)]

    meilleur_global = float('inf')
    perm_optimal = None

    with ProcessPoolExecutor() as workers:
        taches = [workers.submit(traiter_segment, bloc, largeurs, poids) for bloc in sous_ensembles]
        for futur in taches:
            score, permutation = futur.result()
            if score < meilleur_global:
                meilleur_global = score
                perm_optimal = permutation

    return meilleur_global, perm_optimal


if __name__ == "__main__":
    chemin = "EX1/Y-10_t.txt"  
    try:
        nb, larg, pds = charger_instance(chemin)
        print("\nLargeurs lues :")
        print(larg)

        print("\nRecherche parallèle exhaustive :")
        resultat_cout, resultat_perm = recherche_parallele(nb, larg, pds)
        print(f"Optimal cost : {resultat_cout}")
        print(f"Optimale Permutation: {resultat_perm}")
    except FileNotFoundError:
        pass
