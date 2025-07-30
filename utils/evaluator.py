# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:54:34 2024

@author: mlndao
"""
# -*- coding: utf-8 -*-
"""
Fonctions de validation pour évaluer les méthodes d'explicabilité.
Créé par : mlndao
"""

# === Imports ===
import os
import numpy as np
from tqdm import tqdm
from methods import * 
from scipy.stats import spearmanr
import stability_forlder as st

# === Fonctions de validation ===

def identity(X_dist, E_dist):
    """
    Identité : deux objets identiques doivent produire des explications identiques.
    """
    i_dm = X_dist.values
    l_e_dm = E_dist.values
    erreurs = []

    for column in X_dist:
        for row in X_dist:
            if i_dm[row, column] == 0:
                erreurs.append(1 if l_e_dm[row, column] == 0 else 0)
    return np.nanmean(erreurs)


def separability(X_dist, E_dist):
    """
    Séparabilité : deux objets différents doivent produire des explications différentes.
    """
    i_dm = X_dist.values
    l_e_dm = E_dist.values
    erreurs = []

    for column in X_dist:
        for row in X_dist:
            if i_dm[row, column] > 0:
                erreurs.append(1 if l_e_dm[row, column] > 0 else 0)
    return np.nanmean(erreurs)


def stability(X_dist, E_dist):
    """
    Stabilité : objets similaires doivent produire des explications similaires.
    Corrélation de Spearman entre les distances des entrées et des explications.
    """
    rhos = []
    for column in X_dist:
        rho = spearmanr(X_dist.iloc[:, column], E_dist.iloc[:, column])[0]
        rhos.append(rho)

    erreurs = [1 if r >= 0 else 0 for r in rhos]
    return np.nanmean(erreurs)


def coherence(model, explainer, samples, targets, e, nstd=1.5, top_features=5, verbose=True, L2X=False):
    """
    Cohérence : suppression des variables importantes doit dégrader la performance.
    """
    explains = []
    valides = []

    for i in range(len(samples)):
        xi = samples[i:i+1]
        exp = explainer(xi, e, L2X).values
        if np.isnan(exp).any():
            continue

        seuil = exp.mean() + nstd * exp.std()
        nb_important = min((exp > seuil).sum(), top_features)
        aux = exp.flatten()
        seuil = aux[np.argsort(aux)][-nb_important]
        indices = np.argwhere(aux < seuil)
        exp[aux < seuil] = 0

        xic = np.copy(xi).flatten()
        xic[indices] = 0
        explains.append(xic.reshape(xi.shape))
        valides.append(i)

    samples = samples[valides]
    targets = targets[valides]

    tmax = targets.max()
    targets = targets / tmax
    pred = model(samples) / tmax
    erreurs = (pred.reshape(targets.shape) - targets) ** 2

    explains = np.array(explains).reshape(samples.shape)
    pred_exp = model(explains) / tmax
    erreurs_exp = (pred_exp - targets) ** 2

    coh = np.mean(np.abs(erreurs - erreurs_exp))
    compl = min(np.mean(erreurs_exp / erreurs), np.mean(erreurs / erreurs_exp))
    congr = np.sqrt(np.mean((np.abs(erreurs - erreurs_exp) - coh) ** 2))

    return coh, compl, congr


def selectivity(model, explainer, samples, e_x, L2X=False, chunk=1):
    """
    Sélectivité : suppression progressive des variables classées importantes doit augmenter l'erreur.
    """
    scores = []
    for xi in samples:
        exp = explainer(xi[None, ...], e_x, L2X).values
        if np.isnan(exp).any():
            continue

        idxs = exp.flatten().argsort()[::-1]
        xi = xi[0]
        xprime = xi.flatten()
        l = len(xprime)
        blocs = np.split(idxs, int(l / chunk)) if chunk >= 1 else [idxs]

        xs = [xi]
        for bloc in blocs:
            xprime[bloc] = 0
            xs.append(xprime.reshape(xi.shape))
            xprime = np.copy(xprime)

        preds = model(np.array(xs)).flatten()
        e = np.abs(preds[1:] - preds[:-1]) / (np.abs(preds[0]) + 1e-12)
        auc = 1 - (np.cumsum(e) / (e.max() + 1e-12))
        scores.append(1 - np.mean(auc))

    return np.nanmean(scores)


def acumen(explainer, samples, e, L2X=False, nstd=1.5, top_features=5):
    """
    Acuité : vérifie que les points importants perturbés dégradent plus l'explication que les autres.
    """
    rangs = []
    for xi in samples:
        base_exp = explainer(xi[None, ...], e, L2X).values
        if np.isnan(base_exp).any():
            continue

        seuil = base_exp.mean() + nstd * base_exp.std()
        nb = min((base_exp > seuil).sum(), top_features)
        aux = base_exp.flatten()
        top_mask = aux.argsort()[-nb:]

        tc = xi.copy().reshape(base_exp.shape)
        tc[base_exp >= seuil] = 0
        tc_exp = explainer(tc.reshape(xi.shape), e, L2X).values

        aux_tc = tc_exp.flatten()
        rang = (1 - (np.argsort(aux_tc).argsort()[top_mask] / aux_tc.argsort().max())).mean()
        rangs.append(rang)

    return np.nanmean(rangs)

def stability_Velmurugan(explainer, samples, e, top_features=5, verbose=True, L2X=False, nb_iter=5):
    """
    Évalue la stabilité des explications à partir de plusieurs itérations sur les mêmes données.
    Basé sur la mesure de Velmurugan et al., via un score binaire pour les features les plus importantes.

    Paramètres :
    - explainer : fonction générant les explications locales
    - samples : échantillons de données d'entrée
    - e : objet expliquant (ex : KernelSHAP)
    - top_features : nombre de variables les plus importantes à conserver
    - verbose : affichage des étapes (non utilisé ici)
    - L2X : booléen indiquant l’usage d’un explainer type L2X
    - nb_iter : nombre d’itérations pour chaque individu

    Retour :
    - Moyenne des scores de stabilité
    """
    len_flatten = samples.shape[1] * samples.shape[2]
    list_stab_values = []

    for ind in range(samples.shape[0]):
        Z = []
        for _ in range(nb_iter):
            xi = samples[ind:ind+1]
            exp = explainer(xi, e=e, L2X=L2X)
            exp_abs = np.abs(exp).values.flatten()
            threshold = np.sort(exp_abs)[-top_features]
            Zi = [1 if w >= threshold else 0 for w in exp_abs]
            Z.append(Zi)
        stab_value = st.getStability(Z)
        list_stab_values.append(stab_value)

    return np.mean(list_stab_values)


def fidelity_1(model, explainer, samples, e, verbose=True, L2X=False, nb_iter=5, nb_feature=10):
    """
    Calcule la fidélité des explications en mesurant l’impact des caractéristiques
    jugées non importantes sur la prédiction via du bruit aléatoire.

    Paramètres :
    - model : modèle de prédiction
    - explainer : fonction d’explication locale
    - samples : données d’entrée
    - e : objet expliquant (ex : SHAP, LIME)
    - nb_iter : nombre de répétitions
    - nb_feature : nombre de variables les plus importantes conservées
    - L2X : utilisation de L2X ou non

    Retour :
    - Moyenne du MAPE entre prédictions réelles et bruitées
    """
    Y, Y_bar = [], []
    for ind in tqdm(range(samples.shape[0])):
        xi = samples[ind:ind+1]
        for _ in range(nb_iter):
            exp = explainer(xi, e=e, L2X=L2X)
            exp_abs = np.abs(exp).values.flatten()
            threshold = np.sort(exp_abs)[-nb_feature]

            x_bar = xi.flatten()
            for i, w in enumerate(exp_abs):
                if w <= threshold:
                    x_bar[i] += np.random.normal()
            x_bar = x_bar.reshape(xi.shape)

            Y.append(model(xi).values)
            Y_bar.append(model(x_bar).values)

    return compute_MAP(np.array(Y), np.array(Y_bar))


def fidelity(model, explainer, samples, e, nstd=1.5, top_features=10, verbose=True, L2X=False, rd=0, n_iter=10):
    """
    Mesure la fidélité d’un explainer à travers la variation de la prédiction
    suite à la suppression des caractéristiques jugées peu importantes.

    Paramètres :
    - model : fonction de prédiction
    - explainer : méthode d’explication
    - samples : données d’entrée
    - e : explainer (objet)
    - nstd : nombre d’écarts-types pour le seuil
    - top_features : nombre de caractéristiques conservées
    - n_iter : nombre d’itérations
    - L2X : booléen indiquant l’usage de L2X

    Retour :
    - Moyenne du MAPE et proportion des variables modifiées
    """
    score_ = []
    for i in range(len(samples)):
        y, y_pred = [], []
        for k in range(n_iter):
            np.random.seed(k)
            xi = samples[i:i+1]
            exp = np.abs(explainer(xi, e, L2X).values.flatten())
            _mean, _std = exp.mean(), exp.std()
            threshold = np.sort(exp)[-min((exp > (_mean + nstd * _std)).sum(), top_features)]
            x_bar = xi.flatten()
            x_bar[exp <= threshold] += np.random.normal()
            x_bar = x_bar.reshape(xi.shape)
            y.append(model(xi))
            y_pred.append(model(x_bar))
        score_.append(compute_MAPE(np.array(y), np.array(y_pred)))

    M_tot = samples.shape[1] * samples.shape[2]
    return np.mean(score_), min(top_features, M_tot) / M_tot


def instability(model, explainer, samples, e, verbose=True, L2X=False, rd=0, n_iter=5):
    """
    Évalue l’instabilité d’un explainer face à de légères perturbations de l’entrée.

    Paramètres :
    - model : fonction de prédiction
    - explainer : méthode d’explication locale
    - samples : données d’entrée
    - e : explainer objet
    - n_iter : nombre d’itérations de bruitage
    - L2X : booléen (utilise L2X)

    Retour :
    - Moyenne des distances L1 entre explications bruitées et originales
    """
    score_ = []
    for i in range(len(samples)):
        instab = []
        for k in range(n_iter):
            np.random.seed(k)
            xi = samples[i:i+1]
            x_bar = xi + np.random.normal()
            exp_xi = explainer(xi, e, L2X).values
            exp_xi_bar = explainer(x_bar, e, L2X).values
            instab.append(np.linalg.norm(exp_xi - exp_xi_bar, ord=1))
        score_.append(np.mean(instab))

    return np.mean(score_)


def consistency(explainer, samples, e, top_features=5, verbose=True, L2X=False, nb_iter=5):
    """
    Évalue la consistance des explications : les variables les plus importantes 
    doivent rester stables d'une instance à l'autre, si les données sont similaires.

    Paramètres :
    - explainer : fonction d’explication (ex : SHAP, LIME, L2X)
    - samples : tableau d'échantillons d'entrée
    - e : objet expliquant (ex : instance de KernelSHAP)
    - top_features : nombre de variables considérées comme importantes
    - verbose : booléen, pour afficher la progression
    - L2X : booléen, indique si l’explication utilise L2X
    - nb_iter : nombre de répétitions pour la stabilité

    Retour :
    - Moyenne des taux de recouvrement (intersection des top features)
    """
    indices = [i for i in range(samples.shape[0])]
    recouvrement_total = []

    for i in indices:
        top_sets = []
        xi = samples[i:i+1]
        for _ in range(nb_iter):
            exp = explainer(xi, e, L2X).values.flatten()
            if np.isnan(exp).any():
                continue
            top = np.argsort(np.abs(exp))[-top_features:]
            top_sets.append(set(top.tolist()))
        if len(top_sets) > 1:
            base = top_sets[0]
            intersections = [len(base.intersection(other)) / top_features for other in top_sets[1:]]
            recouvrement_total.append(np.mean(intersections))

    return np.mean(recouvrement_total) if recouvrement_total else np.nan


def faithfulness(model, explainer, samples, e, n_features_to_remove=5, L2X=False):
    """
    Évalue la fidélité d'un explainer : l'impact sur la prédiction doit être cohérent 
    avec l'importance des variables retournées par l’explainer.

    Paramètres :
    - model : fonction de prédiction (model.predict)
    - explainer : fonction d’explication
    - samples : échantillons d’entrée
    - e : explainer objet (ex : instance de KernelSHAP)
    - n_features_to_remove : nombre de variables les plus importantes à supprimer
    - L2X : booléen, indique si l’explainer est de type L2X

    Retour :
    - Moyenne des changements relatifs dans les prédictions (score de fidélité)
    """
    variations = []

    for i in range(len(samples)):
        xi = samples[i:i+1]
        exp = explainer(xi, e, L2X).values.flatten()
        if np.isnan(exp).any():
            continue

        top_idx = np.argsort(np.abs(exp))[-n_features_to_remove:]
        x_mod = np.copy(xi).flatten()
        x_mod[top_idx] = 0
        x_mod = x_mod.reshape(xi.shape)

        pred_original = model(xi)
        pred_modified = model(x_mod)
        variation = np.abs(pred_original - pred_modified) / (np.abs(pred_original) + 1e-12)
        variations.append(variation.mean())

    return np.mean(variations) if variations else np.nan
