# from: remove_columns.ipynb

import pandas as pd
import os

directory_path = "archive"

columns_to_remove = [
    "battleTime", # This is a timestamp, which is not useful for the model
    "tournamentTag", # All battles in this dataset are ladder battles
    "arena.id", # All arenas in this dataset are ranked arenas
    "gameMode.id", # All gamemodes in this dataset are ladder battles
]

# Go through each team and add the columns to the columns_to_remove list
teams = ["loser", "winner"]
for team in teams:
    columns = [
        "{}.tag".format(team), # This is a unique identifier for each player (not needed)
        
        "{}.startingTrophies".format(team), # This is not needed since we have the average.startingTrophies
        "{}.trophyChange".format(team),
        "{}.clan.tag".format(team),
        "{}.clan.badgeId".format(team),

        "{}.kingTowerHitPoints".format(team),
        "{}.princessTowersHitPoints".format(team),

        "{}.cards.list".format(team),

        "{}.totalcard.level".format(team),
        "{}.troop.count".format(team),
        "{}.structure.count".format(team),
        "{}.spell.count".format(team),
        "{}.common.count".format(team),
        "{}.rare.count".format(team),
        "{}.epic.count".format(team),
        "{}.legendary.count".format(team),
        "{}.elixir.average".format(team),
    ]
    columns_to_remove.append(
        columns
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

        data.rename({"Unnamed: 0":"index"}, axis="columns", inplace=True)
        data.drop(["index"], axis=1, inplace=True)

        for column in columns_to_remove:
            data.drop(column, inplace=True, axis=1)

        data.to_csv(csv_file, index=False)