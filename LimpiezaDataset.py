from pickle import FALSE, TRUE
import pandas as pd

data=pd.read_csv('spreadspoke_scores.csv')
dataTeams=pd.read_csv('nfl_teams.csv')
# dataStadiums=pd.read_csv('nfl_stadiums.csv')
equiTeams = {}

data=data[['schedule_season','schedule_playoff','team_home','score_home','score_away','team_away','team_favorite_id','spread_favorite']]

data=data.dropna()

# for i in range(0,43):
#     equiTeams[dataTeams['team_name'][i]] = dataTeams['team_id'][i]

data = data.drop(2268,0)

for _, row in dataTeams.iterrows():
    equiTeams[row['team_name']] = row['team_id']

data['team_home'] = data['team_home'].map(equiTeams)
data['team_away'] = data['team_away'].map(equiTeams)

data=data[(data['schedule_playoff']==False)]


print(data.head())