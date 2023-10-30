# from: remove_underlevel.ipynb

import pandas as pd
import os

directory_path = "archive"

columns_to_check_underleveled = []

# Go through each team and each card and add the level column to the list
teams = ["loser", "winner"]
for team in teams:
    for number in range(1, 8 + 1):
        columns_to_check_underleveled.append(
            f"{team}.card{number}.level"
        )

def find_first_csv(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    return None

if os.path.exists(directory_path) and os.path.isdir(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        csv_file = find_first_csv(item_path)
        data = pd.read_csv(csv_file)

        # Keep the match if the levels of all the cards from both players are the same 
        removed_underlevel = data[data[columns_to_check_underleveled].nunique(axis=1) == 1]

        removed_underlevel.to_csv('removed_underlevel.csv', index=False)