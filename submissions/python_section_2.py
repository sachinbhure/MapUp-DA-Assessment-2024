import pandas as pd
import numpy as p
from datetime import datetime, time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Create an empty DataFrame with the same dimensions
    result_df = pd.DataFrame(index=df.index, columns=df.index)

  # Fill the diagonal with zeros
    result_df.values[:] = 0

  # Fill the rest of the matrix using cumulative distances
    for i, row in df.iterrows():
      for j, col in df.iterrows():
        if i != j:
        # Calculate cumulative distance using existing values
          result_df.loc[i, j] = result_df.loc[i, :].sum() - result_df.loc[i, i] + df.loc[i, j]

  # Ensure symmetry
    result_df = result_df.add(result_df.T).div(2)

    return result_df      
    


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    ids = df.index.unique()

  # Create an empty DataFrame
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

  # Iterate through each ID combination
    for i in ids:
      for j in ids:
        if i != j:
          unrolled_df = unrolled_df.append({'id_start': i, 'id_end': j, 'distance': df.loc[i, j]}, ignore_index=True)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_rows = df[df['id_start'] == reference_id]

  # Calculate the average distance for the reference ID
    average_distance = reference_rows['distance'].mean()

  # Calculate   
    threshold = average_distance * 0.1

  # Filter the DataFrame for rows within the threshold
    result_df = df[(df['distance'] >= average_distance - threshold) & (df['distance'] <= average_distance + threshold)]

  # Extract unique 'id_start' values and sort them
    ids_within_threshold = sorted(result_df['id_start'].unique())

    return ids_within_threshold
    


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

  # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
      df[vehicle_type] = df['distance'] * rate_coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    time_ranges = [
      (time(0, 0), time(10, 0), 0.8),
      (time(10, 0), time(18, 0), 1.2),
      (time(18, 0), time(23, 59, 59), 0.8)
    ]
    weekday_discount_factors = {
      'Monday': time_ranges,
      'Tuesday': time_ranges,
      'Wednesday': time_ranges,
      'Thursday': time_ranges,
      'Friday': time_ranges
    }
    weekend_discount_factor = 0.7

  # Create a list of days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  # Create a new DataFrame to store the results
    result_df = pd.DataFrame(columns=df.columns + ['start_day', 'start_time', 'end_day', 'end_time'])

  # Iterate through each unique (id_start, id_end) pair
    for _, row in df.groupby(['id_start', 'id_end']).first().iterrows():
      for day in days_of_week:
        for start_time, end_time, discount_factor in (weekday_discount_factors.get(day) or [('00:00:00', '23:59:59', weekend_discount_factor)]):
        # Create a new row with the appropriate values
          new_row = row.copy()
          new_row['start_day'] = day
          new_row['start_time'] = datetime.strptime(start_time, '%H:%M:%S').time()
          new_row['end_day'] = day if start_time < end_time else days_of_week[(days_of_week.index(day) + 1) % 7]
          new_row['end_time'] = datetime.strptime(end_time, '%H:%M:%S').time()
          new_row['moto'] *= discount_factor
          new_row['car'] *= discount_factor
          new_row['rv'] *= discount_factor
          new_row['bus'] *= discount_factor
          new_row['truck'] *= discount_factor
          result_df = result_df.append(new_row, ignore_index=True)

    return result_df
