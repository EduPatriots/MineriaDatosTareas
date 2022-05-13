from pickle import FALSE, TRUE
import pandas as pd
from requests import head
from tabulate import tabulate
from statsmodels.stats.outliers_influence import summary_table
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode


data=pd.read_csv('spreadspoke_scores.csv')
dataTeams=pd.read_csv('nfl_teams.csv')
# dataStadiums=pd.read_csv('nfl_stadiums.csv')
equiTeams = {}

data=data[['schedule_season','schedule_week','schedule_playoff','team_home','score_home','score_away','team_away','team_favorite_id','spread_favorite']]

data=data.dropna()

data = data.drop(2268,0)

for _, row in dataTeams.iterrows():
    equiTeams[row['team_name']] = row['team_id']

data['team_home'] = data['team_home'].map(equiTeams)
data['team_away'] = data['team_away'].map(equiTeams)

temporadaRegular=data[(data['schedule_playoff']==False)]
handicap_prom = temporadaRegular.agg({"spread_favorite":['mean']})
disparidad_anual = temporadaRegular[['schedule_season','spread_favorite']].groupby(pd.Grouper(key="schedule_season")).mean()


#Aquí empieza la tarea 8

campeones = data[(data['schedule_week']=="Superbowl")]
# campeones = campeones[['schedule_season','spread_favorite']].groupby(pd.Grouper(key="schedule_season"))
condiciones = [campeones['score_home']>campeones['score_away'], campeones['score_home']<campeones['score_away']]
opciones = [campeones['team_home'],campeones['team_away']]
campeones['ganador'] = np.select(condiciones, opciones, "imposible")
campeones = campeones[['schedule_season', 'ganador']]

definitivo = pd.merge(left=disparidad_anual, right=campeones, left_on='schedule_season', right_on='schedule_season')
# for _, row in data.iterrows():
#     equiTeams[row['team_name']] = row['team_id']
#     if(data[row['schedule_playoff']] == "Superbowl"):
#         campeones.add()

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))


def normalize_distribution(dist: np.array, n: int) -> np.array:
    b = dist - min(dist) + 0.000001
    c = (b / np.sum(b)) * n
    return np.round(c)


def create_distribution(mean: float, size: int) -> pd.Series:
    return normalize_distribution(np.random.standard_normal(size), mean * size)


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    plt.savefig(file_path)
    plt.close()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(
    points: List[np.array], labels: np.array, input_data: List[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]

# groups = definitivo
# df = generate_df(groups, 50)
df = definitivo
print(df)
df = df[(df['ganador']=="NE")|(df['ganador']=="PIT")|(df['ganador']=="SF")|(df['ganador']=="DAL")|(df['ganador']=="NYG")]
scatter_group_by("Vecino_Mas_Cercano", df, "schedule_season", "spread_favorite", "ganador")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5,
)
print(kn)