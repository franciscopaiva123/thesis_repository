import pickle

import pandas as pd
from keras import Model
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Permute, Concatenate
from keras.src.metrics import  Precision
from keras.src.optimizers import Adam
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, BatchNormalization

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from preprocess import preprocess_data
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalMaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42
            )
""" Step 1: Load the CSV Data """
file_path = "DATASETS/fixed_gear.csv"
dataset_name = os.path.splitext(os.path.basename(file_path))[0]

# Read the CSV file into a pandas DataFrame
dataset = pd.read_csv(file_path)

# Drop unnecessary columns
dataset = dataset.drop(columns=['distance_from_shore'])
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], unit='s')

# Convert timestamp to datetime format


""" Step 2: Pre-Process the Data"""


# Remove outliers and impossible values
dataset_clean = preprocess_data(dataset)
print("Signal Fishing:", len(dataset_clean[(dataset_clean['is_fishing'] == 1.0)]))
print("Signal not Fishing:", len(dataset_clean[(dataset_clean['is_fishing'] == 0.0)]))
print("All Signals:", len(dataset_clean))

def normalize(df):
    # Create a copy
    df_normalized = df.copy()

    # Columns to exclude from normalization
    exclude_cols = ['mmsi', 'timestamp']

    # Get features to normalize (all columns except excluded ones)
    features_to_normalize = [col for col in df.columns if col not in exclude_cols]

    # Group by vessel ID (mmsi) and normalize each vessel's data separately
    for mmsi, vessel_data in df.groupby('mmsi'):
        # Create a temporary DataFrame for this vessel
        vessel_df = vessel_data.copy()

        # Process each feature
        for feature in features_to_normalize:
            # Handle infinities
            vessel_df[feature] = vessel_df[feature].replace([np.inf, -np.inf], np.nan)

            # Handle NaNs using forward and backward fill
            vessel_df[feature] = vessel_df[feature].fillna(method='bfill').fillna(method='ffill')

            # If still NaN, fill with zeros
            if vessel_df[feature].isna().any():
                vessel_df[feature] = vessel_df[feature].fillna(0)

            # Check if there's variation in the feature
            if vessel_df[feature].nunique() > 1:
                # Initialize a scaler for this feature and vessel
                scaler = MinMaxScaler()

                # Normalize the feature
                values = vessel_df[feature].values.reshape(-1, 1)
                vessel_df[feature] = scaler.fit_transform(values).flatten()
            else:
                # If all values are the same, set to 0 (or 0.5 to indicate middle value)
                vessel_df[feature] = 0.5 if vessel_df[feature].iloc[0] != 0 else 0

        # Update the normalized DataFrame with this vessel's normalized data
        df_normalized.loc[vessel_df.index] = vessel_df

    return df_normalized



def feature_engineer(segment_df):

    # Create a copy
    df = segment_df.copy()
    df = df[['mmsi', 'timestamp', 'lat', 'lon', 'speed', 'course','is_fishing']]

    # If the segment is too small, return
    if len(df) < 3:
        return df

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Extract arrays for feature computation
    timestamps = pd.to_datetime(df['timestamp'])
    speeds = df['speed'].values
    courses = df['course'].values
    lats = df['lat'].values
    lons = df['lon'].values

    # == "NORMAL" FEATURES ==

    df['time_diff'] = (timestamps.diff().fillna(pd.Timedelta(seconds=0)) / pd.Timedelta(seconds=1)).astype(float)

    df['time_of_day'] = timestamps.dt.hour + timestamps.dt.minute / 60

    df['speed_diff'] = df['speed'].diff().fillna(0)

    # Rolling window for statistics (adjust window size as needed)
    window_size = min(5, len(df))

    # Speed statistics
    df['speed_mean'] = df['speed'].rolling(window=window_size, min_periods=1).mean()
    df['speed_std'] = df['speed'].rolling(window=window_size, min_periods=1).std().fillna(0)
    df['speed_mean_squared'] = df['speed_mean'] ** 2
    df['speed_std_squared'] = df['speed_std'] ** 2

    # Acceleration statistics
    df['acceleration'] = df['speed_diff'] / df['time_diff'].replace(0, np.nan).fillna(1)
    df['acceleration'] = df['acceleration'].fillna(0)
    df['acceleration_mean'] = df['acceleration'].rolling(window=window_size, min_periods=1).mean()
    df['acceleration_std'] = df['acceleration'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Jerk
    df['jerk'] = df['acceleration'].diff().fillna(0) / df['time_diff'].replace(0, np.nan).fillna(1)
    df['jerk'] = df['jerk'].fillna(0)
    df['jerk_mean'] = df['jerk'].rolling(window=window_size, min_periods=1).mean()
    df['jerk_std'] = df['jerk'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Course differences
    course_diff = df['course'].diff().fillna(0)
    course_diff = np.minimum(np.abs(course_diff), 360 - np.abs(course_diff))
    df['course_diff'] = course_diff

    # Course statistics
    df['course_mean'] = course_circular_mean(df['course'].rolling(window=window_size, min_periods=1))
    df['course_std'] = course_circular_std(df['course'].rolling(window=window_size, min_periods=1))
    df['course_mean_squared'] = df['course_mean'] ** 2
    df['course_std_squared'] = df['course_std'] ** 2

    # Lat/Lon differences
    df['lat_diff'] = df['lat'].diff().fillna(0)
    df['lon_diff'] = df['lon'].diff().fillna(0)

    # Lat/Lon statistics
    df['lat_diff_mean'] = df['lat_diff'].rolling(window=window_size, min_periods=1).mean()
    df['lat_diff_std'] = df['lat_diff'].rolling(window=window_size, min_periods=1).std().fillna(0)
    df['lon_diff_mean'] = df['lon_diff'].rolling(window=window_size, min_periods=1).mean()
    df['lon_diff_std'] = df['lon_diff'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Distance calculations
    distances = calculate_distances(lats, lons)
    df['distance'] = distances
    df['distance_mean'] = df['distance'].rolling(window=window_size, min_periods=1).mean()
    df['distance_std'] = df['distance'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Distance over speed (time to travel distance at current speed)
    df['distance_over_speed'] = df['distance'] / df['speed'].replace(0, np.nan).fillna(0.1)
    df['distance_over_speed_mean'] = df['distance_over_speed'].rolling(window=window_size, min_periods=1).mean()
    df['distance_over_speed_std'] = df['distance_over_speed'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Covariance and autocorrelation
    df['speed_course_cov'] = calculate_rolling_covariance(df['speed'], df['course'], window_size)

    df['speed_autocorr'] = calculate_autocorrelation(df['speed'], 1)

    df['course_autocorr'] = calculate_autocorrelation(df['course'], 1)

    # ==FISHING-SPECIFIC FEATURES==

    # CIRCULARITY and AREA (especially for purse seines)

    # Start-to-end distance
    start_end_dist = haversine_distance(lats[0], lons[0], lats[-1], lons[-1])

    total_path_length = df['distance'].sum()

    # Circularity
    circularity = start_end_dist / total_path_length if total_path_length > 0 else 1
    df['circularity'] = circularity

    # Area covered
    try:
        hull = ConvexHull(np.column_stack([lons, lats]))
        area_covered = hull.volume
        df['area_covered'] = area_covered
    except:
        df['area_covered'] = 0

    # PATH LINEARITY FEATURES

    # Path linearity aka ratio of straight-line distance to total path length
    df['path_linearity'] = start_end_dist / total_path_length if total_path_length > 0 else 0

    # 3. SPEED PATTERN FEATURES

    # Time spent at different fishing-relevant speeds
    df['time_at_trawl_speed'] = np.mean((speeds >= 2) & (speeds <= 5))
    df['time_at_seine_speed'] = np.mean(speeds <= 2)
    df['time_stopped'] = np.mean(speeds < 0.5)

    # Speed phase analysis
    if len(speeds) >= 6:
        # Split into thirds
        seg_size = len(speeds) // 3
        first_third = speeds[:seg_size]
        middle_third = speeds[seg_size:2 * seg_size]
        last_third = speeds[2 * seg_size:]

        first_third_speed = np.mean(first_third)
        middle_third_speed = np.mean(middle_third)
        last_third_speed = np.mean(last_third)

        df['first_third_speed'] = first_third_speed
        df['middle_third_speed'] = middle_third_speed
        df['last_third_speed'] = last_third_speed

        # Speed phase pattern:
        # longlines: moderate -> slow -> moderate
        longline_pattern = 0
        if 2 < first_third_speed < 6 and middle_third_speed < 1 and 1 < last_third_speed < 4:
            longline_pattern = 1

        # purse seines: fast -> slow -> very slow
        seine_pattern = 0
        if first_third_speed > 5 and middle_third_speed < 3 and last_third_speed < 1:
            seine_pattern = 1

        # trawlers: consistent moderate speed
        trawl_pattern = 0
        speeds_std = np.std([first_third_speed, middle_third_speed, last_third_speed])
        if 2 < np.mean([first_third_speed, middle_third_speed, last_third_speed]) < 5 and speeds_std < 1:
            trawl_pattern = 1

        # Combine into a categorical feature
        speed_pattern = 0  # default: no distinct pattern
        if longline_pattern:
            speed_pattern = 1
        elif seine_pattern:
            speed_pattern = 2
        elif trawl_pattern:
            speed_pattern = 3

        df['speed_phase_pattern'] = speed_pattern
    else:
        df['first_third_speed'] = 0
        df['middle_third_speed'] = 0
        df['last_third_speed'] = 0
        df['speed_phase_pattern'] = 0

    # COURSE PATTERN FEATURES

    # Count significant turns
    segment_duration_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600
    if segment_duration_hours > 0:
        sharp_turns = np.sum(course_diff > 30)
        df['sharp_turns_per_hour'] = sharp_turns / segment_duration_hours
    else:
        df['sharp_turns_per_hour'] = 0

    # STOP-AND-GO PATTERNS

    # Transitions from moving to stopped or vice versa
    stopped_mask = speeds < 0.5
    transitions = np.sum(np.abs(np.diff(stopped_mask.astype(int))))

    if segment_duration_hours > 0:
        df['stop_go_transitions_per_hour'] = transitions / segment_duration_hours
    else:
        df['stop_go_transitions_per_hour'] = 0

    return df


# Helper functions

def haversine_distance(lat1, lon1, lat2, lon2):
    # Haversine distance between two points in kms
    R = 6371  # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
        dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def calculate_distances(lats, lons):
    # Distances between consecutive points using Haversine formula
    distances = [0]  # First point has no previous point
    for i in range(1, len(lats)):
        dist = haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
        distances.append(dist)
    return distances


def course_circular_mean(rolling_series):
    # Calculate circular mean of course values
    result = []
    for window in rolling_series:
        if len(window) == 0:
            result.append(0)
            continue
        sin_sum = np.sum(np.sin(np.radians(window)))
        cos_sum = np.sum(np.cos(np.radians(window)))
        circular_mean = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
        result.append(circular_mean)
    return np.array(result)


def course_circular_std(rolling_series):
    # Calculate circular standard deviation of course values
    result = []
    for window in rolling_series:
        if len(window) <= 1:
            result.append(0)
            continue
        sin_vals = np.sin(np.radians(window))
        cos_vals = np.cos(np.radians(window))
        sin_mean = np.mean(sin_vals)
        cos_mean = np.mean(cos_vals)
        r = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
        circular_std = np.sqrt(-2 * np.log(r))
        result.append(np.degrees(circular_std))
    return np.array(result)


def calculate_rolling_covariance(series1, series2, window_size):
    # Calculate rolling covariance
    cov_values = []
    for i in range(len(series1)):
        start_idx = max(0, i - window_size + 1)
        s1_window = series1.iloc[start_idx:i + 1]
        s2_window = series2.iloc[start_idx:i + 1]
        if len(s1_window) > 1:
            cov = np.cov(s1_window, s2_window)[0, 1]
        else:
            cov = 0
        cov_values.append(cov)
    return np.array(cov_values)


def calculate_autocorrelation(series, lag):
    # Calculate lag-N autocorrelation
    result = np.zeros(len(series))
    for i in range(lag, len(series)):
        window = series.iloc[i - lag:i + 1]
        if len(window) >= lag + 1:
            try:
                # Calculate autocorrelation with lag
                corr, _ = pearsonr(window.iloc[:-lag], window.iloc[lag:])
                result[i] = corr
            except:
                result[i] = 0
    return result


def feature_engineer_by_vessel(df):
    df_engineered = pd.DataFrame()
    # Process each vessel independently

    for mmsi, vessel_data in df.groupby('mmsi'):
        # Apply feature engineering to this vessel
        vessel_df = feature_engineer(vessel_data)
        df_engineered = pd.concat([df_engineered, vessel_df], ignore_index=True)

    return df_engineered


"""Step 4: Concatenate TimeSequences"""


def segment_vessel_data(df, segment_length=45, max_time_hours=1, min_valid_signals=7, label=''):
    all_segments = []

    for mmsi, vessel_data in df.groupby('mmsi'):
        vessel_data = vessel_data.sort_values('timestamp').reset_index(drop=True)

        i = 0
        while i < len(vessel_data):
            start_time = vessel_data.loc[i, 'timestamp']
            segment = []
            activity_label = vessel_data.loc[
                i, 'is_fishing']

            # Accumulate up to 100 signals within a 4-hour period
            while i < len(vessel_data):
                current_time = vessel_data.loc[i, 'timestamp']
                time_diff_hours = (current_time - start_time) / np.timedelta64(1,
                                                                               'h')  # Convert time difference to hours
                if time_diff_hours > max_time_hours or len(segment) >= segment_length:
                    break

                # Append features for this signal
                segment.append([
                    vessel_data.iloc[i]['speed'],
                    vessel_data.iloc[i]['course'],
                    vessel_data.iloc[i]['time_diff'],
                    vessel_data.iloc[i]['time_of_day'],
                    vessel_data.iloc[i]['speed_diff'],
                    vessel_data.iloc[i]['speed_mean'],
                    vessel_data.iloc[i]['speed_std'],
                    vessel_data.iloc[i]['speed_mean_squared'],
                    vessel_data.iloc[i]['speed_std_squared'],
                    vessel_data.iloc[i]['acceleration'],
                    vessel_data.iloc[i]['acceleration_mean'],
                    vessel_data.iloc[i]['acceleration_std'],
                    vessel_data.iloc[i]['jerk'],
                    vessel_data.iloc[i]['jerk_mean'],
                    vessel_data.iloc[i]['jerk_std'],
                    vessel_data.iloc[i]['course_diff'],
                    vessel_data.iloc[i]['course_mean'],
                    vessel_data.iloc[i]['course_std'],
                    vessel_data.iloc[i]['course_mean_squared'],
                    vessel_data.iloc[i]['course_std_squared'],
                    vessel_data.iloc[i]['lat_diff'],
                    vessel_data.iloc[i]['lat_diff_mean'],
                    vessel_data.iloc[i]['lat_diff_std'],
                    vessel_data.iloc[i]['lon_diff'],
                    vessel_data.iloc[i]['lon_diff_mean'],
                    vessel_data.iloc[i]['lon_diff_std'],
                    vessel_data.iloc[i]['distance'],
                    vessel_data.iloc[i]['distance_mean'],
                    vessel_data.iloc[i]['distance_std'],
                    vessel_data.iloc[i]['distance_over_speed'],
                    vessel_data.iloc[i]['distance_over_speed_mean'],
                    vessel_data.iloc[i]['distance_over_speed_std'],
                    vessel_data.iloc[i]['speed_course_cov'],
                    vessel_data.iloc[i]['speed_autocorr'],
                    vessel_data.iloc[i]['course_autocorr'],
                    vessel_data.iloc[i]['circularity'],
                    vessel_data.iloc[i]['area_covered'],
                    vessel_data.iloc[i]['path_linearity'],
                    vessel_data.iloc[i]['time_at_trawl_speed'],
                    vessel_data.iloc[i]['time_at_seine_speed'],
                    vessel_data.iloc[i]['time_stopped'],
                    vessel_data.iloc[i]['first_third_speed'],
                    vessel_data.iloc[i]['middle_third_speed'],
                    vessel_data.iloc[i]['last_third_speed'],
                    vessel_data.iloc[i]['speed_phase_pattern'],
                    vessel_data.iloc[i]['sharp_turns_per_hour'],
                    vessel_data.iloc[i]['stop_go_transitions_per_hour']
                ])
                i += 1

            # Zero pad to ensure each segment has exactly 25 signals
            segment_padded = pad_sequences([segment], maxlen=segment_length, padding='post', dtype='float32')[0]
            # Check if the segment has at least `min_valid_signals` non-padded rows
            valid_signals_count = np.count_nonzero(np.any(segment_padded != 0, axis=1))
            if valid_signals_count >= min_valid_signals:
                all_segments.append((segment_padded, activity_label))

    return all_segments


dataset_clean_1 = feature_engineer_by_vessel(dataset_clean)
dataset_normalized_1 = normalize(dataset_clean_1)
segmented_data_0 = segment_vessel_data(
    dataset_normalized_1[dataset_normalized_1['is_fishing'] == 0.0].reset_index(drop=True))
segmented_data_1 = segment_vessel_data(
    dataset_normalized_1[dataset_normalized_1['is_fishing'] == 1.0].reset_index(drop=True))
segmented_datax = segmented_data_0 + segmented_data_1

print("First Segment", segmented_datax[0][0])
print("First Segment Label ", segmented_datax[0][1])

print("First Segment", segmented_datax[-1][0])
print("First Segment Label ", segmented_datax[-1][1])

# BUILD CNN

"""Step 5: Prepare input for the CNN"""

X = []
y = []

for seq, label in segmented_datax:
    X.append(seq)
    y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

# Reshape X to be 3D: (n_samples, total_time_steps, n_features)
n_features = X.shape[2]  # Number of features has increased with engineered features
total_time_steps = X.shape[1]  # Total time steps (concatenated time segments)

# Split into train/test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

# Define the model

input_layer = Input(shape=(total_time_steps, n_features))

# === CNN BRANCH ===
# First Conv block
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu',padding='same')(input_layer)
conv1 = Dropout(0.1)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = MaxPooling1D(pool_size=2)(conv1)

# Second Conv block
conv2 = Conv1D(filters=128, kernel_size=3, activation='relu',padding='same')(conv1)
conv2 = Dropout(0.2)(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = MaxPooling1D(pool_size=2)(conv2)


cnn_flat = Flatten()(conv2)

# === LSTM BRANCH ===
# Dimension shuffle (transpose time and features)
lstm_input = Permute((2, 1))(input_layer)

# LSTM with dropout
lstm_out = LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(lstm_input)

lstm_out_flat = Flatten()(lstm_out)

# === CONCATENATE BOTH BRANCHES ===
concat = Concatenate()([cnn_flat, lstm_out_flat])
# For binary classification (fishing or not fishing)
output_layer = Dense(1, activation='sigmoid')(concat)
model = Model(inputs=input_layer, outputs=output_layer)


# Compile the model
cp1 = ModelCheckpoint('model1.keras', save_best_only=True)
model.compile(optimizer=Adam(learning_rate=0.00002), loss='binary_crossentropy',
              metrics=['accuracy', Precision()])
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=15,  # Number of epochs to wait before stopping if no improvement
    restore_best_weights=True  # Restore the weights of the best epoch
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    validation_data=(X_train, y_train),
    epochs=80,
    callbacks=[cp1, early_stopping]
)

# Save the model


model.save("CNN_LSTM.keras")

print(history.history)
# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)

# Convert probabilities to binary predictions with a threshold
threshold = 0.5  # Experiment with different thresholds
y_pred = (y_pred_prob > threshold).astype(int)

# Print metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print F1 score
# For overall F1 score, use one of the following:
f1_micro = f1_score(y_test, y_pred, average='micro')  # Global metric
f1_macro = f1_score(y_test, y_pred, average='macro')  # Unweighted average
f1_weighted = f1_score(y_test, y_pred, average='weighted')  # Weighted average

print("F1 (micro):", f1_micro)
print("F1 (macro):", f1_macro)
print("F1 (weighted):", f1_weighted)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix for better visualization
cm_normalized = cm.astype('float')

# Define class labels
labels = ["Not Fishing", "Fishing"]

# Plot normalized confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
disp.plot(cmap='viridis')

# Add a title
plt.title("Normalized Confusion Matrix")
plt.show()


