import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pbp = pd.read_csv('/Users/benbroner/nfllive/data/multiline5.csv')
oneline = pd.read_csv('/Users/benbroner/nfllive/data/nflds1.csv')
playerteam = pd.read_csv('/Users/benbroner/nfllive/data/playerteam.csv')


def oneline_win(df = playerteam):
	# defines and trains a oneline game model
	win_model = GradientBoostingClassifier()
	mlp = MLPClassifier()
	dtc = DecisionTreeClassifier()
	data = df
	columns = data.columns
	player_cols = list(columns[188:])
	all_columns = ['a_elo_pre', 'h_elo_pre', 'Home_team_gen_avg_rush_yards', 'Away_team_gen_avg_rush_yards', 'Away_team_gen_avg_penalties', 'Home_team_gen_avg_penalties', 'Away_team_gen_avg_turnovers', 'Home_team_gen_avg_turnovers', 'Away_team_gen_avg_pass_yards', 'Home_team_gen_avg_pass_yards', 'Home_team_gen_avg_time_of_possesion', 'Away_team_gen_avg_time_of_possesion', 'Home_team_gen_avg_score', 'Away_team_gen_avg_score', 'Away_team_gen_avg_points_allowed', 'Home_team_gen_avg_points_allowed', 'h_win'] + player_cols
	train_columns = ['a_elo_pre', 'h_elo_pre', 'Home_team_gen_avg_rush_yards', 'Away_team_gen_avg_rush_yards', 'Away_team_gen_avg_penalties', 'Home_team_gen_avg_penalties', 'Away_team_gen_avg_turnovers', 'Home_team_gen_avg_turnovers', 'Away_team_gen_avg_pass_yards', 'Home_team_gen_avg_pass_yards', 'Home_team_gen_avg_time_of_possesion', 'Away_team_gen_avg_time_of_possesion', 'Home_team_gen_avg_score', 'Away_team_gen_avg_score', 'Away_team_gen_avg_points_allowed', 'Home_team_gen_avg_points_allowed'] +player_cols
	df = data[all_columns]
	print(df.shape)
	df = df.dropna()
	x = df[train_columns + player_cols]
	x = StandardScaler().fit_transform(x)

	y = df[['h_win']]

	win_model.fit(x, y)
	mlp.fit(x, y)
	dtc.fit(x, y)
	# win_model = CalibratedClassifierCV(win_model,cv=5,method='sigmoid')
	h = win_model.fit(x, y)

	print(cross_val_score(win_model, x, y, cv=5))
	# print(cross_val_score(mlp, x, y, cv=5))
	print(cross_val_score(dtc, x, y, cv=5))
	return win_model

def pbp_win(df=pbp):
	# defines and trains a playby play model
	win_model = GradientBoostingClassifier()
	data = df
	all_columns = ['a_elo_pre', 'h_elo_pre', 'Home_team_gen_avg_rush_yards', 'Away_team_gen_avg_rush_yards', 'Away_team_gen_avg_penalties', 'Home_team_gen_avg_penalties', 'Away_team_gen_avg_turnovers', 'Home_team_gen_avg_turnovers', 'Away_team_gen_avg_pass_yards', 'Home_team_gen_avg_pass_yards', 'Home_team_gen_avg_time_of_possesion', 'Away_team_gen_avg_time_of_possesion', 'Home_team_gen_avg_score', 'Away_team_gen_avg_score', 'Away_team_gen_avg_points_allowed', 'Home_team_gen_avg_points_allowed', 'ball', 'yards', 'Down', 'ToGo', 'Quarter', 'pg_spread', 'h_win', 'ranker']
	train_columns = ['a_elo_pre', 'h_elo_pre', 'Home_team_gen_avg_rush_yards', 'Away_team_gen_avg_rush_yards', 'Away_team_gen_avg_penalties', 'Home_team_gen_avg_penalties', 'Away_team_gen_avg_turnovers', 'Home_team_gen_avg_turnovers', 'Away_team_gen_avg_pass_yards', 'Home_team_gen_avg_pass_yards', 'Home_team_gen_avg_time_of_possesion', 'Away_team_gen_avg_time_of_possesion', 'Home_team_gen_avg_score', 'Away_team_gen_avg_score', 'Away_team_gen_avg_points_allowed', 'Home_team_gen_avg_points_allowed', 'ball', 'yards', 'Down', 'ToGo', 'pg_spread', 'ranker']
	df = data[all_columns]
	print(df.shape)
	df = df[~df.yards.str.contains("na")]
	df = df.dropna()
	x = df[train_columns]
	y = df[['h_win']]

	win_model.fit(x, y)
	print(cross_val_score(win_model, x, y, cv=5))
	return win_model

def pbp_totals(df = pbp):
	# constantly predicts totals
	totals_model = GradientBoostingRegressor()

	data = df
	all_columns = ['a_elo_pre', 'h_elo_pre', 'Home_team_gen_avg_rush_yards', 'Away_team_gen_avg_rush_yards', 'Away_team_gen_avg_penalties', 'Home_team_gen_avg_penalties', 'Away_team_gen_avg_turnovers', 'Home_team_gen_avg_turnovers', 'Away_team_gen_avg_pass_yards', 'Home_team_gen_avg_pass_yards', 'Home_team_gen_avg_time_of_possesion', 'Away_team_gen_avg_time_of_possesion', 'Home_team_gen_avg_score', 'Away_team_gen_avg_score', 'Away_team_gen_avg_points_allowed', 'Home_team_gen_avg_points_allowed', 'h_win', 'h_total', 'a_total']
	train_columns = ['a_elo_pre', 'h_elo_pre', 'Home_team_gen_avg_rush_yards', 'Away_team_gen_avg_rush_yards', 'Away_team_gen_avg_penalties', 'Home_team_gen_avg_penalties', 'Away_team_gen_avg_turnovers', 'Home_team_gen_avg_turnovers', 'Away_team_gen_avg_pass_yards', 'Home_team_gen_avg_pass_yards', 'Home_team_gen_avg_time_of_possesion', 'Away_team_gen_avg_time_of_possesion', 'Home_team_gen_avg_score', 'Away_team_gen_avg_score', 'Away_team_gen_avg_points_allowed', 'Home_team_gen_avg_points_allowed']
	df = data[all_columns]
	print(df.shape)
	df = df.dropna()
	x = df[train_columns]
	x = StandardScaler().fit_transform(x)

	y = df[['h_total']]

	h_team = totals_model.fit(x, y)
	print((cross_val_score(h_team, x, y, cv=5)))
	y = df[['a_total']]
	a_team = totals_model.fit(x, y)
	
	print(cross_val_score(a_team, x, y, cv=5))

	# win_model = CalibratedClassifierCV(win_model,cv=5,method='sigmoid')

	# print(cross_val_score(mlp, x, y, cv=5))
oneline_win(playerteam)


