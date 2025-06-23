DEFAULT_SAMPLING_RATE = 100  # Fréquence d'échantillonnage en Hz

def plot_signal_pronostia(df, signal_name, unit=None):
    #     train = df
    plt.figure(figsize=(13,5))
    if unit:
        plt.plot('RUL_norm', signal_name,
                data=df[df['Unit']==unit])
    else:
        for i in df['Unit'].unique():
            # if (i % 10 == 0):  # only ploting every 10th unit_nr
            plt.plot('RUL_norm', signal_name, data=df[df['Unit']==i])
            
    plt.xlim(2560, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 2560, 250))
    plt.ylabel(signal_name)
    plt.xlabel('Remaining Use fulLife')
    #plt.savefig(signal_name+'.jpeg')
    plt.show()

# Chargement des données
def load_vibration_data(data_folder, idx=None):
    """Charge les fichiers de vibration et calcule le temps en secondes pour chaque échantillon."""
    vibration_files = [f for f in os.listdir(data_folder) if f.startswith('acc_') and f.endswith('.csv')]
    if not vibration_files:
        print("No vibration files found in the specified folder.")
        return pd.DataFrame()  # Retourne un DataFrame vide si aucun fichier n'est trouvé
    
    vibration_data = []

    for file in vibration_files:
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path, names=['Hour', 'Minute', 'Second', 'Microsecond', 'Horizontal_Accel', 'Vertical_Accel'])
        
        # Extraction de l'identifiant du roulement
        bearing_id = file.split('_')[1].split('.')[0]
        df['bearing_id'] = idx + bearing_id
        
        # Calcul du temps en secondes pour chaque échantillon
        df['time_seconds'] = df['Hour'] * 3600 + df['Minute'] * 60 + df['Second'] + df['Microsecond'] * 1e-6
        vibration_data.append(df)

    # Concaténation de toutes les données de vibration en un DataFrame unique
    result = pd.concat(vibration_data, ignore_index=True)
    print(f"Loaded data shape: {result.shape}")
    print(result.head())
    return result

# Calcul du temps total et du RUL normalisé
def calculate_total_time(vibration_data):
    """Calcule le temps total de l'expérience pour chaque roulement."""
    return vibration_data.groupby('bearing_id')['time_seconds'].max()

def calculate_normalized_rul(vibration_data, total_time):
    """Calcule le RUL normalisé pour chaque échantillon."""
    vibration_data['RUL_norm'] = vibration_data.apply(
        lambda row: 1 - (total_time[row['bearing_id']] - row['time_seconds']) / total_time[row['bearing_id']],
        axis=1
    )
    return vibration_data

# Division en fenêtres
def split_into_windows(data, window_size, sampling_rate=DEFAULT_SAMPLING_RATE):
    """Divise les données en fenêtres de taille définie."""
    samples_per_window = int(window_size * sampling_rate)
    num_windows = len(data) // samples_per_window
    print(f"Data size: {len(data)}, Samples per window: {samples_per_window}, Number of windows: {num_windows}")
    if num_windows == 0:
        print("Warning: Not enough data to form a window.")
    windows = [data[i * samples_per_window:(i + 1) * samples_per_window] for i in range(num_windows)]
    return windows

# Extraction des caractéristiques temporelles
def extract_temporal_features(signal):
    """Extrait les caractéristiques temporelles."""
    features = {
        'mean': signal.mean(),
        'std': signal.std(),
        'peak_to_peak': signal.max() - signal.min(),
        'rms': np.sqrt(np.mean(signal**2)),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
    }
    return features

# Extraction des caractéristiques fréquentielles
def extract_frequency_features(signal, sampling_rate=DEFAULT_SAMPLING_RATE):
    """Extrait les caractéristiques fréquentielles."""
    N = len(signal)
    freqs = fft(signal)
    freqs = np.abs(freqs[:N // 2])
    freq_bins = np.fft.fftfreq(N, d=1 / sampling_rate)[:N // 2]
    
    features = {
        'max_amplitude': np.max(freqs),
        'mean_freq': np.mean(freqs),
        'rms_freq': np.sqrt(np.mean(freqs**2)),
    }
    return features

# Extraction de toutes les caractéristiques
def extract_features_per_window(data, window_size, sampling_rate=DEFAULT_SAMPLING_RATE):
    """Extrait les caractéristiques de chaque fenêtre de données."""
    features_list = []

    for bearing_id, group in data.groupby('bearing_id'):
        print(f"Processing bearing: {bearing_id}, Data size: {len(group)}")
        horizontal_signal = group['Horizontal_Accel'].values
        vertical_signal = group['Vertical_Accel'].values
        
        horizontal_windows = split_into_windows(horizontal_signal, window_size, sampling_rate)
        vertical_windows = split_into_windows(vertical_signal, window_size, sampling_rate)
        
        total_duration = group['time_seconds'].max()
        
        for i, (h_window, v_window) in enumerate(zip(horizontal_windows, vertical_windows)):
            temporal_features_h = extract_temporal_features(h_window)
            temporal_features_v = extract_temporal_features(v_window)
            frequency_features_h = extract_frequency_features(h_window)
            frequency_features_v = extract_frequency_features(v_window)
            
            features = {f"{k}_h": v for k, v in temporal_features_h.items()}
            features.update({f"{k}_v": v for k, v in temporal_features_v.items()})
            features.update({f"{k}_freq_h": v for k, v in frequency_features_h.items()})
            features.update({f"{k}_freq_v": v for k, v in frequency_features_v.items()})
            
            # Calculer le RUL normalisé pour cette fenêtre
            start_time_window = i * window_size  # Temps de début de la fenêtre en secondes
            RUL = max(0, total_duration - start_time_window)  # RUL décroît avec le temps
            RUL_norm = RUL / total_duration if total_duration > 0 else 0  # RUL normalisé
            
            # Ajouter des informations contextuelles
            features['bearing_id'] = bearing_id
            features['window_index'] = i
            features['RUL_norm'] = RUL_norm  # Ajout de RUL normalisé
            
            features_list.append(features)

    return pd.DataFrame(features_list)

# Pipeline principal
def main_pipeline(data_folder, idx, window_size=10, sampling_rate=DEFAULT_SAMPLING_RATE):
    """Pipeline complet pour traiter les données."""
    # Charger les données
    vibration_data = load_vibration_data(data_folder, idx)
    if vibration_data.empty:
        print("No data loaded. Check your data folder.")
        return pd.DataFrame()
    
    # Calculer le temps total et le RUL normalisé
    total_time = calculate_total_time(vibration_data)
    vibration_data = calculate_normalized_rul(vibration_data, total_time)
    print(f"Data after RUL calculation: {vibration_data.shape}")
    
    # Extraire les caractéristiques
    features_df = extract_features_per_window(vibration_data, window_size, sampling_rate)
    print(f"Extracted features shape: {features_df.shape}")
    
    return features_df
