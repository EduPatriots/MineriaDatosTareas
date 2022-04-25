from pickle import FALSE, TRUE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import numbers
import statsmodels.api as sm


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
# print(handicap_prom)

max_handicap = data.agg({"spread_favorite":['idxmin']})

max_handicap = max_handicap['spread_favorite'].astype(str).astype(int)

# print(data.loc[int(max_handicap)], "\nLa apuesta con mayor handicap en la historia la ganó quien aposto por el contrario\n")

min_handicap = data.agg({"spread_favorite":['idxmax']})

min_handicap = min_handicap['spread_favorite'].astype(str).astype(int)

# print(data.loc[int(min_handicap)], "\nSorprendentemente existió una apuesta al empate a pesar de la rareza de estos en la liga")

#Aquí empieza la tarea 4

disparidad_anual = data[['schedule_season','spread_favorite']].groupby(pd.Grouper(key="schedule_season")).mean()

# plt.plot(disparidad_anual)
# plt.show()

#Aquí empieza la tarea 5

#Hipotesis: La casa siempre gana

casa_gana = 0
casa_pierde = 0

for _,i in data.iterrows():
    dif = abs(i[3]-i[4])
    handicap = abs(i[7])
    if(i[3]>i[4]):
        winner = i[2]
    else:
        winner = i[5]

    if(dif>handicap and winner == i[6]):
        casa_gana += 1
    else:
        casa_pierde += 1

# print(casa_gana, casa_pierde)

# if(casa_gana>casa_pierde):
#     print("Hipotesis confirmada, la casa siempre gana")
# else:
#     print("Hipotesis falsa")

#Aquí empieza la tarea 6

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x:str, y: str)->None:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='green')
    plt.plot(df_by_sal[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='red')
    plt.xticks(rotation=90)
    plt.savefig(f'lr_{y}_{x}.png')
    plt.close()

df_by_sal = disparidad_anual
df_by_sal.reset_index(inplace=True)
print_tabulate(df_by_sal.head(5))
linear_regression(df_by_sal, "schedule_season", "spread_favorite")