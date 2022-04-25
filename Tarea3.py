from pickle import FALSE, TRUE
import pandas as pd

data=pd.read_csv('spreadspoke_scores.csv')
dataTeams=pd.read_csv('nfl_teams.csv')
# dataStadiums=pd.read_csv('nfl_stadiums.csv')
equiTeams = {}

data=data[['schedule_season','schedule_playoff','team_home','score_home','score_away','team_away','team_favorite_id','spread_favorite']]

data=data.dropna()

data = data.drop(2268,0)

for _, row in dataTeams.iterrows():
    equiTeams[row['team_name']] = row['team_id']

data['team_home'] = data['team_home'].map(equiTeams)
data['team_away'] = data['team_away'].map(equiTeams)

data=data[(data['schedule_playoff']==False)]

#Aquí empieza la tarea 3

handicap_prom = data.agg({"spread_favorite":['mean']})
print(handicap_prom)

max_handicap = data.agg({"spread_favorite":['idxmin']})

max_handicap = max_handicap['spread_favorite'].astype(str).astype(int)

print(data.loc[int(max_handicap)], "\nLa apuesta con mayor handicap en la historia la ganó quien aposto por el contrario\n")

min_handicap = data.agg({"spread_favorite":['idxmax']})

min_handicap = min_handicap['spread_favorite'].astype(str).astype(int)

print(data.loc[int(min_handicap)], "\nSorprendentemente existió una apuesta al empate a pesar de la rareza de estos en la liga")