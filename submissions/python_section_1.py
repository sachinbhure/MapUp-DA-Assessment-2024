from typing import Dict, List, Any, Tuple
import re,math, polyline
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.append(lst[i + n - 1 - j])
        
        result.extend(group)
    
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    result = {}

    # Group strings by their length
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []  # Initialize list for this length
        result[length].append(string)

    # Sort the dictionary by key (length) in ascending order
    return dict(sorted(result.items()))

    

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def _flatten(current_dict: Any, parent_key: str = '') -> Dict:
        items = []

        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                items.extend(_flatten(value, new_key).items())
            elif isinstance(value, list):
                # Handle lists by iterating through elements
                for i, elem in enumerate(value):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(elem, dict):
                        # Recursively flatten nested dictionaries inside lists
                        items.extend(_flatten(elem, list_key).items())
                    else:
                        # Add non-dict elements directly
                        items.append((list_key, elem))
            else:
                # Add other elements directly
                items.append((new_key, value))

        return dict(items)

    return _flatten(nested_dict)
    

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(path, used):
        # If the current path is the same length as the input list, it's a complete permutation
        if len(path) == len(nums):
            result.append(path[:])  # Add a copy of the current path to the results
            return
        
        for i in range(len(nums)):
            # Skip used elements or duplicate elements (only take the first occurrence in each recursion level)
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            # Mark the current element as used and add it to the path
            used[i] = True
            path.append(nums[i])
            
            # Recursively generate the next permutation
            backtrack(path, used)
            
            # Backtrack by removing the last element and marking it as unused
            path.pop()
            used[i] = False

    # Sort the input list to ensure duplicates are adjacent
    nums.sort()
    result = []
    backtrack([], [False] * len(nums))
    return result
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.

    """
    date_pattern = r'\b(\d{2}-\d{2}-\d{4})\b|\b(\d{2}/\d{2}/\d{4})\b|\b(\d{4}\.\d{2}\.\d{2})\b'
    
    # Find all matches using the regex
    matches = re.findall(date_pattern, text)
    
    # Flatten the matches and remove empty strings
    valid_dates = [match for group in matches for match in group if match]
    
    return valid_dates
    pass

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two coordinates in meters.
    """
    R = 6371000  # Radius of the Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates: List[Tuple[float, float]] = polyline.decode(polyline_str)

    # Create a DataFrame with latitude and longitude
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Initialize the distance column with 0 for the first row
    df['distance'] = 0.0

    # Calculate distances using the Haversine formula
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)

    return df
    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here

    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated = [[0] * n for _ in range(n)]  # Initialize an empty n x n matrix
    for i in range(n):
        for j in range(n):
            rotated[j][n - i - 1] = matrix[i][j]

    # Step 2: Replace each element with the sum of its row and column elements (excluding itself)
    final_matrix = [[0] * n for _ in range(n)]  # Initialize the transformed matrix
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column, excluding the current element
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix
    


def time_check(df:pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a multi-index based on (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    def is_complete(group):
        # Generate a date range for the week for each day from min to max date
        all_days = pd.date_range(start=group['start'].min().date(), end=group['end'].max().date(), freq='D')

        # Check if all 7 days of the week are present
        week_days = all_days.weekday
        unique_week_days = set(week_days)

        if len(unique_week_days) != 7:
            return True  # Incomplete if not all 7 days are present

        # Check for 24-hour coverage for each day
        for day in all_days:
            day_data = group[(group['start'].dt.date == day.date()) | (group['end'].dt.date == day.date())]
            if not (day_data['start'].dt.time.min() <= pd.to_datetime('00:00:00').time() and
                    day_data['end'].dt.time.max() >= pd.to_datetime('23:59:59').time()):
                return True  # Incomplete if 24-hour period is not covered for any day

        return False  # Complete if all checks are passed

    # Apply the completeness check for each group and return a boolean series
    completeness_results = grouped.apply(is_complete)
    return completeness_results
    
