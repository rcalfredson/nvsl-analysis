def parse_distances(distances_str):
    try:
        # Convert the string to a list of floats
        distances = list(map(float, distances_str.split(",")))

        # Check if the list is monotonically increasing
        if all(x < y for x, y in zip(distances, distances[1:])):
            return distances
        else:
            raise ValueError("The distances must be in monotonically increasing order.")
    except ValueError as e:
        print(f"Error: {e}")
        raise
