# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:32:25 2025

@author: mlndao
"""

# === Importations standard ===
from __future__ import print_function
import os
import sys
import math
import random

# === Bibliothèques scientifiques ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# === Machine Learning et Deep Learning ===
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Layer
from tensorflow.keras import backend as K
from keras import backend as K  # Keras backend (compatible avec TF)
import keras

# === Outils de modélisation et d’évaluation ===
import sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# === Statistiques, distance, optimisation ===
from scipy import optimize
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform, euclidean

# === Interprétabilité et segmentation de signal ===
from lime.lime_tabular import RecurrentTabularExplainer
from tqdm import tqdm
from itertools import combinations
import ruptures as rpt


# === Fonctions de traitement des données ===

def prepare_data(file_name):
    """
    Prépare les données d'entraînement et de test à partir des fichiers texte.
    """
    dir_path = '../data/'
    index_names = ['Unit', 'Cycle']
    setting_names = ['Altitude', 'Mach', 'TRA']
    sensor_names = ['T20','T24','T30','T50','P20','P15','P30','Nf','Nc','epr','Ps30','phi',
                    'NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
    col_names = index_names + setting_names + sensor_names

    df_train = pd.read_csv(dir_path + 'train_' + str(file_name), delim_whitespace=True, names=col_names)
    rul_train = pd.DataFrame(df_train.groupby('Unit')['Cycle'].max()).reset_index()
    rul_train.columns = ['Unit', 'max']
    df_train = df_train.merge(rul_train, on=['Unit'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    df_test = pd.read_csv(dir_path + 'test_' + str(file_name), delim_whitespace=True, names=col_names)
    y_test = pd.read_csv(dir_path + 'RUL_' + str(file_name), delim_whitespace=True, names=["RUL"])

    return df_train, df_test, y_test


def prep_data(train, test, drop_sensors, remaining_sensors, alpha, drop=True):
    """
    Prépare les données en ajoutant les conditions de fonctionnement, 
    en les normalisant, puis en lissant les signaux.
    """
    if drop:
        X_train = add_operating_condition(train.drop(drop_sensors, axis=1))
        X_test = add_operating_condition(test.drop(drop_sensors, axis=1))
    else:
        X_train = add_operating_condition(train)
        X_test = add_operating_condition(test)

    X_train, X_test = condition_scaler(X_train, X_test, remaining_sensors)
    X_train = exponential_smoothing(X_train, remaining_sensors, 0, alpha)
    X_test = exponential_smoothing(X_test, remaining_sensors, 0, alpha)

    return X_train, X_test


def add_operating_condition(df):
    """
    Ajoute une variable catégorielle représentant la condition de fonctionnement.
    """
    df = df.copy()
    df['Altitude'] = df['Altitude'].round()
    df['Mach'] = df['Mach'].round(2)
    df['TRA'] = df['TRA'].round()

    df['op_cond'] = df['Altitude'].astype(str) + '_' + df['Mach'].astype(str) + '_' + df['TRA'].astype(str)
    return df


def condition_scaler(df_train, df_test, sensor_names):
    """
    Applique un MinMaxScaler à chaque condition de fonctionnement distincte.
    """
    scaler = MinMaxScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def plot_signal(df, signal_name, unit=None):
    """
    Affiche un graphique de l’évolution d’un capteur en fonction de la RUL.
    """
    plt.figure(figsize=(13, 5))
    if unit:
        plt.plot('RUL', signal_name, data=df[df['Unit'] == unit])
    else:
        for i in df['Unit'].unique():
            if i % 10 == 0:
                plt.plot('RUL', signal_name, data=df[df['Unit'] == i])
    plt.xlim(350, 0)
    plt.xticks(np.arange(0, 375, 25))
    plt.ylabel(signal_name)
    plt.xlabel('Remaining Useful Life')
    plt.show()


def exponential_smoothing(df, sensors, n_samples, alpha=0.3):
    """
    Lisse les signaux des capteurs via une moyenne exponentielle.
    """
    df = df.copy()
    new_column = df.groupby('Unit')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())
    df[sensors] = new_column.reset_index(level=0, drop=True)

    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('Unit')['Unit'].transform(create_mask, samples=n_samples).astype(bool)
    return df[mask]


def root_mean_squared_error(y_true, y_pred):
    """
    Calcule la racine de l’erreur quadratique moyenne.
    """
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def compute_s_score(rul_true, rul_pred):
    """
    Calcule la métrique S-score définie dans l’article original.
    """
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))


def evaluate(y_true, y_hat, label='test'):
    """
    Affiche les métriques RMSE et R² pour les prédictions fournies.
    """
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print(f'{label} set RMSE: {rmse}, R2: {variance}')


def gen_train_data(df, sequence_length, columns):
    """
    Génère les séquences d'entraînement (glissantes) à partir des colonnes données.
    """
    data = df[columns].values
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements - (sequence_length - 1)), range(sequence_length, num_elements + 1)):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    """
    Génère les données d'entraînement pour tous les 'unit_nrs' (moteurs).
    """
    if unit_nrs.size <= 0:
        unit_nrs = df['Unit'].unique()

    data_gen = (list(gen_train_data(df[df['Unit'] == unit_nr], sequence_length, columns)) for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array

def create_model(TW , remaining_):
#     history = History()
    model = Sequential()
    model.add(LSTM(units=128, activation='tanh',input_shape=(TW, len(remaining_))))
    model.add(Dense(units=128, activation='relu'))
    #model.add(GlobalAveragePooling1D(name = 'feature_layer'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    return model

def compute_MAPE(y_true, y_hat):
    mape = np.mean(np.abs((y_true - y_hat)/y_true))*100
    return mape

def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['Unit'].unique()

    label_gen = [gen_labels(df[df['Unit']==unit_nr], sequence_length, label)
                for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array

def gen_test_data(df, sequence_length, columns, mask_value):
    """
    Génère une séquence de test : si elle est trop courte, la complète par un masque.
    Retourne uniquement la dernière séquence complète.
    """
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)  # padding
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values
    else:
        data_matrix = df[columns].values

    stop = data_matrix.shape[0]
    start = stop - sequence_length
    yield data_matrix[start:stop, :]


def plot_loss(fit_history):
    """
    Affiche l'évolution de la fonction de perte pour les phases entraînement et validation.
    """
    plt.figure(figsize=(13, 5))
    plt.plot(range(1, len(fit_history.history['loss']) + 1), fit_history.history['loss'], label='train')
    plt.plot(range(1, len(fit_history.history['val_loss']) + 1), fit_history.history['val_loss'], label='validation')
    plt.xlabel('Épochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def new_column(df, column):
    """
    Ajoute une colonne d’index croissants à un DataFrame.
    """
    df[column] = range(1, len(df) + 1)
    return df

def reduce_channels(image, axis=-1, op='sum'):
    """
    Réduit une image multicanaux selon une opération donnée : somme, moyenne ou max absolu.
    """
    if op == 'sum':
        return image.sum(axis=axis)
    elif op == 'mean':
        return image.mean(axis=axis)
    elif op == 'absmax':
        pos_max = image.max(axis=axis)
        neg_max = -((-image).max(axis=axis))
        return np.select([pos_max >= neg_max, pos_max < neg_max], [pos_max, neg_max])


def gamma_correction(image, gamma=0.4, minamp=0, maxamp=None):
    """
    Applique une correction gamma à l’image pour améliorer la visibilité.
    """
    c_image = np.zeros_like(image)
    image -= minamp
    if maxamp is None:
        maxamp = np.abs(image).max() + 1e-7
    image /= maxamp
    pos_mask = (image > 0)
    neg_mask = (image < 0)
    c_image[pos_mask] = np.power(image[pos_mask], gamma)
    c_image[neg_mask] = -np.power(-image[neg_mask], gamma)
    return c_image * maxamp + minamp


def project_image(image, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    """
    Projette une image entre deux bornes (par défaut [0,1]) avec normalisation.
    """
    if absmax is None:
        absmax = np.max(np.abs(image), axis=tuple(range(1, len(image.shape))))
    absmax = np.asarray(absmax)
    mask = (absmax != 0)
    if mask.sum() > 0:
        image[mask] = image[mask] / np.expand_dims(absmax[mask], axis=-1)
    if not input_is_positive_only:
        image = (image + 1) / 2
    image = image.clip(0, 1)
    return output_range[0] + image * (output_range[1] - output_range[0])

def get_model_params(model):
    """
    Récupère les noms de couches, activations, poids et objets couches d’un modèle Keras.
    """
    names, activations, weights, layers = [], [], [], []
    for layer in model.layers:
        names.append(layer.name)
        activations.append(layer.output)
        weights.append(layer.get_weights())
        layers.append(layer)
    return names, activations, weights, layers

def display(
    signal,
    true_chg_pts,
    computed_chg_pts=None,
    computed_chg_pts_color="k",
    computed_chg_pts_linewidth=3,
    computed_chg_pts_linestyle="--",
    computed_chg_pts_alpha=1.0,
    **kwargs
):
    """
    Affiche un signal ainsi que ses points de changement avec mise en couleur des segments.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib est requis pour utiliser cette fonction.")

    if not isinstance(signal, np.ndarray):
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    options_fig = {"figsize": (10, 2 * n_features)}
    options_fig.update(kwargs)
    fig, axarr = plt.subplots(n_features, sharex=True, **options_fig)
    if n_features == 1:
        axarr = [axarr]

    cmap = mpl.colormaps['RdYlGn']
    for i, (axe, sig) in enumerate(zip(axarr, signal.T)):
        axe.plot(range(n_samples), sig)

        for (_, start, end, imp) in true_chg_pts[i]:
            axe.axvspan(max(0, start - 0.5), end - 0.5, color=cmap(imp), alpha=0.4)

        if computed_chg_pts is not None:
            for pt in computed_chg_pts:
                axe.axvline(pt, color=computed_chg_pts_color,
                            linewidth=computed_chg_pts_linewidth,
                            linestyle=computed_chg_pts_linestyle,
                            alpha=computed_chg_pts_alpha)

    fig.tight_layout()
    return fig, axarr

def usegment(signal, n_bkps=2):
    """
    Segmentation uniforme : divise chaque signal en 'n_bkps' segments de taille équivalente.

    Args:
        signal (np.ndarray): Signal multivarié (shape : [n_features, n_timesteps]).
        n_bkps (int): Nombre de segments par signal.

    Returns:
        list: Liste de tuples (feature_index, start, end) représentant les segments.
    """
    period = len(signal[0]) // n_bkps
    result = []
    for i in range(signal.shape[0]):
        result += [(i, p, p + period) for p in range(0, len(signal[0]), period)]
    return result


def segment(signal, n_bkps=2):
    """
    Segmentation par détection de ruptures (méthode dynamique, modèle L2).

    Args:
        signal (np.ndarray): Signal multivarié (shape : [n_features, n_timesteps]).
        n_bkps (int): Nombre de ruptures à détecter par feature.

    Returns:
        list: Liste de tuples (feature_index, start, end) représentant les segments détectés.
    """
    result = []
    for i in range(signal.shape[0]):
        algo = rpt.Dynp(model="l2").fit(signal[i].T)
        r = [0] + algo.predict(n_bkps)
        result += [(i, l, r_) for l, r_ in zip(r, r[1:])]
    return result


def sampling(signal, segments, feature_faker, n=10, k=3):
    """
    Génère des versions masquées du signal (z) et leurs masques associés (zprime).

    Args:
        signal (np.ndarray): Signal à expliquer (shape : [n_features, n_timesteps]).
        segments (list): Liste des segments à perturber (ex: issus de segment()).
        feature_faker (callable): Fonction pour générer des valeurs de remplacement.
        n (int): Nombre total de combinaisons à générer.
        k (int): Nombre maximum de segments modifiés simultanément.

    Returns:
        list: Liste de tuples (zprime, z) où :
              - zprime : masque binaire indiquant les segments visibles,
              - z : version du signal avec certains segments masqués.
    """
    ranges = [(signal[i].min(), signal[i].max()) for i in range(signal.shape[0])]
    mean_std = [(signal[i].mean(), signal[i].std()) for i in range(signal.shape[0])]
    zprimes = []

    # Échantillons déterministes : combinaisons de segments
    for kk in range(1, k):
        for seg in combinations(segments, kk):
            zprime = [1 if s in seg else 0 for s in segments]
            z = np.copy(signal)
            for zj, (i, start, end) in zip(zprime, segments):
                if zj == 0:
                    z[i, start:end] = feature_faker(*ranges[i], *mean_std[i], end - start)
            zprimes.append((zprime, z))
            if len(zprimes) >= n:
                return zprimes

    # Échantillons aléatoires restants
    while len(zprimes) < n:
        seg = random.sample(segments, random.randint(1, len(segments)-1))
        zprime = [1 if s in seg else 0 for s in segments]
        z = np.copy(signal)
        for zj, (i, start, end) in zip(zprime, segments):
            if zj == 0:
                z[i, start:end] = feature_faker(*ranges[i], *mean_std[i], end - start)
        zprimes.append((zprime, z))

    return zprimes

def mean_sample(signal):
    """
    Crée un échantillon moyen à partir d’un signal en remplaçant chaque valeur
    par la moyenne de la série correspondante (par variable).

    Args:
        signal (np.ndarray): Signal original (shape : [n_features, n_timesteps]).

    Returns:
        np.ndarray: Signal moyen simulé.
    """
    result = np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        result[i] = signal[i].mean()
    return result

class Sample_Concrete(Layer):
    """
    Couche personnalisée Keras pour l'échantillonnage de variables discrètes
    via le mécanisme Gumbel-Softmax (utilisé dans L2X pour le masquage différentiable).

    Args:
        tau0 (float): Température initiale du Gumbel-Softmax.
        k (int): Nombre de caractéristiques à sélectionner.
    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        """
        Génère les échantillons différentiables (entraînement) ou discrets (inférence).
        """
        logits_ = K.expand_dims(logits, -2)
        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape=(batch_size, self.k, d),
                                    minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)
        gumbel = -K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Seuil discret (inférence)
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'tau0': self.tau0,
            'k': self.k
        })
        return config

def validate_acumen(explainer, samples, iterations=100, nstd=1.5, top_features=100, verbose=True):
    """
    Évalue le pouvoir discriminant de l'explication en perturbant les caractéristiques les plus importantes
    et en comparant leur impact à des perturbations aléatoires.

    Args:
        explainer: Objet expliquant les prédictions.
        samples: Échantillons à tester.
        iterations: Nombre d'itérations.
        nstd: Seuil basé sur la moyenne + n * écart-type.
        top_features: Nombre maximal de caractéristiques importantes.
        verbose: Affichage de la progression.

    Returns:
        Moyenne des scores de rang (plus proche de 1 est meilleur).
    """
    ranking = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1]
        base_exp = explainer.explain(xi)
        if not np.isnan(base_exp).any():
            _mean, _std = base_exp.mean(), base_exp.std()
            threshold = _mean + nstd * _std
            nsamples = min((base_exp > threshold).sum(), top_features)
            aux1 = base_exp.flatten()
            top_mask = aux1.argsort()[-nsamples:]

            tc = np.copy(xi).reshape(base_exp.shape)
            tc[base_exp >= threshold] = 0
            tc_exp = explainer.explain(tc.reshape(xi.shape))
            aux = tc_exp.flatten()
            aux_max = aux.argsort().max()

            score = (1 - (np.argsort(aux).argsort()[top_mask] / aux_max)).mean()
            ranking.append(score)
    return np.nanmean(ranking)

def validate_coherence(model, explainer, samples, targets, nstd=1.5, top_features=100, verbose=True):
    """
    Évalue la cohérence des explications : supprime les caractéristiques importantes et 
    vérifie si la performance du modèle diminue significativement.

    Args :
        model : modèle de prédiction.
        explainer : objet expliquant les prédictions.
        samples : échantillons d'entrée (X).
        targets : cibles réelles (y).
        nstd : seuil pour détecter les caractéristiques importantes.
        top_features : nombre maximal de caractéristiques importantes à considérer.
        verbose : affiche la progression.

    Retourne :
        dict avec les métriques : coherence, completeness, congruency.
    """

    explains, valid_idx = [], []

    for i in tqdm.tqdm(range(len(samples)), disable=not verbose):
        xi = samples[i:i+1]
        exp = explainer.explain(xi)

        if not np.isnan(exp).any():
            _mean, _std = exp.mean(), exp.std()
            threshold = _mean + nstd * _std
            nsamples = min((exp > threshold).sum(), top_features)

            aux = exp.flatten()
            threshold = aux[aux.argsort()][-nsamples]
            indexes = np.argwhere(aux < threshold)

            exp[aux < threshold] = 0
            xic = np.copy(xi).flatten()
            xic[indexes] = 0
            explains.append(xic.reshape(xi.shape))
            valid_idx.append(i)

    samples = samples[valid_idx]
    targets = targets[valid_idx]
    explains = np.array(explains).reshape(samples.shape)

    tmax = targets.max()
    targets = targets / tmax

    pred = model.predict(samples) / tmax
    errors = 1 - (pred.reshape(targets.shape) - targets) ** 2

    exp_pred = model.predict(explains) / tmax
    exp_errors = 1 - (exp_pred.reshape(targets.shape) - targets) ** 2

    coherence_i = np.abs(errors - exp_errors)
    coherence = np.mean(coherence_i)

    return {
        'coherence': coherence,
        'completeness': np.mean(exp_errors / errors),
        'congruency': np.sqrt(np.mean((coherence_i - coherence) ** 2))
    }


def validate_identity(model, explainer, samples, verbose=True):
    """
    Vérifie le principe d'identité : deux appels à explain sur le même échantillon doivent
    produire la même explication.

    Retourne la proportion d'exemples respectant cette propriété.
    """

    errors = []
    for i in tqdm.tqdm(range(samples.shape[0]), disable=not verbose):
        xi = samples[i:i+1]
        exp_a = explainer.explain(xi)
        exp_b = explainer.explain(xi)

        if not np.isnan(exp_a).any() and not np.isnan(exp_b).any():
            errors.append(1 if np.all(exp_a == exp_b) else 0)

    return np.nanmean(errors)

def validate_separability(model, explainer, samples, verbose=True):
    """
    Vérifie que deux échantillons différents ont des explications différentes.

    Retourne une moyenne binaire : 1 si les explications sont différentes, 0 sinon.
    """


    explains, valid_samples = [], []

    for xi in tqdm.tqdm(samples, disable=not verbose):
        exp = explainer.explain(xi[np.newaxis])
        if not np.isnan(exp).any():
            explains.append(exp)
            valid_samples.append(xi)

    explains = np.array(explains)
    valid_samples = np.array(valid_samples)

    scores = []
    for i in range(len(valid_samples) - 1):
        for j in range(i + 1, len(valid_samples)):
            if not np.array_equal(valid_samples[i], valid_samples[j]):
                dist = np.sum((explains[i] - explains[j]) ** 2)
                scores.append(1 if dist > 0 else 0)

    return np.nanmean(scores)

def validate_stability(model, explainer, samples, verbose=True):
    """
    Évalue la stabilité des explications vis-à-vis de la proximité des données d’entrée.

    Retourne la moyenne des corrélations de Spearman entre les distances des entrées
    et les distances des explications.
    """

    explains, valid_samples = [], []

    for xi in tqdm.tqdm(samples, disable=not verbose):
        exp = explainer.explain(xi[np.newaxis])
        if not np.isnan(exp).any():
            explains.append(exp)
            valid_samples.append(xi)

    explains = np.array(explains)
    valid_samples = np.array(valid_samples)

    correlations = []
    for i in range(len(valid_samples)):
        dxs, des = [], []
        for j in range(len(valid_samples)):
            if i == j:
                continue
            dxs.append(euclidean(valid_samples[i].flatten(), valid_samples[j].flatten()))
            des.append(euclidean(explains[i].flatten(), explains[j].flatten()))
        corr = spearmanr(dxs, des).correlation
        if not np.isnan(corr):
            correlations.append(corr)

    return np.nanmean(correlations)

def validate_selectivity(model, explainer, samples, samples_chunk=1, verbose=True):
    """
    Mesure la sélectivité : suppression séquentielle des caractéristiques les plus importantes
    et observation de la baisse de performance.

    Retourne une mesure de la robustesse du classement des caractéristiques.
    """
    
    scores = []

    for i in tqdm.tqdm(range(len(samples)), disable=not verbose):
        xi = samples[i:i+1]
        exp = explainer.explain(xi)

        if np.isnan(exp).any():
            continue

        xi_flat = xi.flatten()
        indices = exp.flatten().argsort()[::-1]

        if samples_chunk >= 1:
            indices = np.split(indices, len(indices) // samples_chunk)

        x_variants = [xi_flat.copy()]
        for idx_group in indices:
            xi_flat[idx_group] = 0
            x_variants.append(xi_flat.reshape(xi.shape))
            xi_flat = x_variants[0].flatten().copy()

        preds = model.predict(np.array(x_variants)).flatten()
        diffs = np.abs(preds[1:] - preds[:-1]) / (preds[0] + 1e-12)
        auc = np.cumsum(diffs)
        auc = 1 - (auc / (auc.max() + 1e-12))
        score = 1 - np.mean(auc)
        scores.append(score)

    return np.nanmean(scores)
