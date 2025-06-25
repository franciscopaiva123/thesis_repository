import pandas as pd
import numpy as np




# Filter impossible coordinates
def remove_invalid_coordinates(df):
    df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]
    return df


# Function to calculate the distance between two lat/lon points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # Distance in kilometers


# Function to remove teleportation based on distance and time difference
def remove_teleportation(df, max_distance=74.08, max_time_diff=7200):  # max_time_diff in seconds (2 hours)
    # Maximum distance = 2(hours) ×20(knots/h) × 1.852=74.08kilometers
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate the time difference between consecutive rows in seconds
    df['time_diff'] = df.groupby('mmsi')['timestamp'].diff().dt.total_seconds().fillna(0)

    # Calculate the distance between consecutive latitude/longitude points
    df['distance'] = haversine(df['lat'], df['lon'], df['lat'].shift(), df['lon'].shift())

    # Remove rows where:
    # - The distance exceeds max_distance (74.08 km for 2 hours at 20 knots)
    # - The time difference exceeds 2 hours (7200 seconds)
    df = df[(df['distance'] <= max_distance) | (df['time_diff'] > max_time_diff)]

    # Drop temporary columns
    df = df.drop(columns=['time_diff', 'distance'])

    return df


# Filter negative or zero SOG, and excessively high speeds
def filter_sog_outliers(df, max_speed=20):
    df = df[(df['speed'] > 0) & (df['speed'] <= max_speed)]
    return df


# Filter invalid COG values
def filter_invalid_cog(df):
    df = df[df['course'].between(0, 360)]
    return df


def add_noise(data, noise_level=0.01):
    """
    Adds Gaussian noise to time-series data.

    Args:
        data (numpy.ndarray): The input time-series data (3D: samples, time_steps, features).
        noise_level (float): The standard deviation of the noise.

    Returns:
        numpy.ndarray: Augmented data with added noise.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


# Filter sudden/unexplained changes in course
# Errado temos que tomar em consideracao tempo
def remove_rapid_course_changes(df, max_course_change_per_minute=20):
    # df['course_diff'] = df.groupby('mmsi')['course'].diff().fillna(0).abs()
    # df = df[df['course_diff'] <= max_course_change]
    # df = df.drop(columns=['course_diff'])
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate time differences in minutes between consecutive signals
    df['time_diff'] = df.groupby('mmsi')['timestamp'].diff().dt.total_seconds() / 60  # time difference in minutes
    df['time_diff'] = df['time_diff'].fillna(0)

    # Calculate course differences between consecutive signals
    df['course_diff'] = df.groupby('mmsi')['course'].diff().fillna(0).abs()

    # Determine allowable course change based on time difference
    df['allowed_course_change'] = df['time_diff'] * max_course_change_per_minute

    # Keep only rows where the course change is within the allowable range
    df = df[df['course_diff'] <= df['allowed_course_change']]

    # Drop the temporary columns
    df = df.drop(columns=['course_diff', 'time_diff', 'allowed_course_change'])

    return df


# Remove irregular time gaps
def remove_irregular_time_gaps(df, time_gap_minutes=10):
    # Iterate through each vessel
    for mmsi, vessel_data in df.groupby('mmsi'):
        # Sort vessel data by timestamp
        vessel_data = vessel_data.sort_values('timestamp')

        # Compute time differences (in minutes) between consecutive signals
        time_diffs = vessel_data['timestamp'].diff().dt.total_seconds() / 60.0

        # Find indices where time difference exceeds the threshold
        invalid_indices = vessel_data.index[time_diffs > time_gap_minutes]

        # Drop rows with these indices from the original DataFrame
        df.drop(invalid_indices, inplace=True)

    # Reset the index of the updated DataFrame
    df.reset_index(drop=True, inplace=True)





# Filter turn-based outliers: Sharp turns without corresponding speed change
def remove_turn_based_outliers(df, min_turn=45, min_speed_change=2):
    # Calculate the absolute difference in 'course' between consecutive rows for each vessel (mmsi)
    df['course_diff'] = df.groupby('mmsi')['course'].diff().fillna(0).abs()

    # Calculate the absolute difference in 'speed' between consecutive rows for each vessel (mmsi)
    df['speed_diff'] = df.groupby('mmsi')['speed'].diff().fillna(0).abs()

    # Identify rows where there is a sharp turn (course_diff > min_turn)
    df['turn'] = df['course_diff'] > min_turn

    # Filter out rows where there is a sharp turn but the speed change is small (speed_diff < min_speed_change)
    df = df[~((df['turn']) & (df['speed_diff'] < min_speed_change))]

    # Drop temporary columns used for filtering
    df = df.drop(columns=['course_diff', 'speed_diff', 'turn'])

    return df


def preprocess_data(df):
    #df = df[(df['is_fishing'] == 1.0) | (df['is_fishing'] == 0)]
    df = df[(df['is_fishing'] == 1.0)]
    df = df.drop_duplicates(subset=['mmsi', 'timestamp'])
    df = remove_invalid_coordinates(df)
    df = filter_sog_outliers(df)
    df = filter_invalid_cog(df)
    remove_irregular_time_gaps(df)

    # df = remove_teleportation(df)
    #df = remove_rapid_course_changes(df)
    #df = remove_turn_based_outliers(df)
    return df

# Apply the preprocessing to your dataset


