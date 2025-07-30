# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:05:06 2024

@author: mlndao
"""
# === Importation des fonctions utiles ===
from methods import *

# === Prétraitement des données ===
def prep_data(train, test, drop_sensors, remaining_sensors, alpha, drop=True):
    """
    Prépare les données d'entraînement et de test :
    - Supprime certains capteurs si nécessaire
    - Ajoute la condition opérationnelle
    - Applique une normalisation conditionnelle
    - Applique un lissage exponentiel
    
    Paramètres :
        train (DataFrame) : données d'entraînement
        test (DataFrame) : données de test
        drop_sensors (list) : liste des capteurs à supprimer
        remaining_sensors (list) : capteurs conservés pour l'analyse
        alpha (float) : coefficient de lissage exponentiel
        drop (bool) : appliquer ou non la suppression des capteurs

    Retour :
        X_train_interim, X_test_interim (DataFrames) : données transformées
    """
    if drop:
        X_train_interim = add_operating_condition(train.drop(drop_sensors, axis=1))
        X_test_interim = add_operating_condition(test.drop(drop_sensors, axis=1))
    else:
        X_train_interim = add_operating_condition(train)
        X_test_interim = add_operating_condition(test)

    X_train_interim, X_test_interim = condition_scaler(X_train_interim, X_test_interim, remaining_sensors)
    X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, 0, alpha)
    X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, 0, alpha)
    
    return X_train_interim, X_test_interim

# === Fonction de transformation RUL ===
def rul_piecewise_fct(X_train, rul):
    """
    Applique une coupure à la variable RUL pour limiter la valeur maximale.

    Paramètres :
        X_train (DataFrame) : données d'entraînement contenant une colonne 'RUL'
        rul (int) : valeur maximale de RUL autorisée

    Retour :
        DataFrame avec la RUL tronquée à la valeur spécifiée
    """
    X_train['RUL'].clip(upper=rul, inplace=True)
    return X_train

