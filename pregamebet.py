from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

def yo(odd):
    # to find the adjusted odds multiplier 
    # returns float
    if odd == 0:
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)

feature_cols = ['a_ML', 'h_ML', 'elo_h_pre',
       'elo_a_pre', 'pitcherhera', 'pitcherhpperi', 'pitcherhwhip',
       'pitcheraera', 'pitcherapperi', 'pitcherawhip', 'firsthba', 'firsthobp',
       'firsthops', 'secondhba', 'secondhobp', 'secondhops', 'thirdhba',
       'thirdhobp', 'thirdhops', 'fourthhba', 'fourthhobp', 'fourthhops',
       'fifthhba', 'fifthhobp', 'fifthhops', 'sixthhba', 'sixthhobp',
       'sixthhops', 'seventhhba', 'seventhhobp', 'seventhhops', 'eigthhba',
       'eigthhobp', 'eigthhops', 'ninthhba', 'ninthhobp', 'ninthhops',
       'firstaba', 'firstaobp', 'firstaops', 'secondaba', 'secondaobp',
       'secondaops', 'thirdaba', 'thirdaobp', 'thirdaops', 'fourthaba',
       'fourthaobp', 'fourthaops', 'fifthaba', 'fifthaobp', 'fifthaops',
       'sixthaba', 'sixthaobp', 'sixthaops', 'seventhaba', 'seventhaobp',
       'seventhaops', 'eigthaba', 'eigthaobp', 'eigthaops', 'ninthaba',
       'ninthaobp', 'ninthaops']

new_feature_cols = ['a_ML', 'h_ML', 'h_elo_pre',
       'a_elo_pre', 'pitcherhera', 'pitcherhpperi', 'pitcherhwhip',
       'pitcheraera', 'pitcherapperi', 'pitcherawhip', 'firsthba', 'firsthobp',
       'firsthops', 'secondhba', 'secondhobp', 'secondhops', 'thirdhba',
       'thirdhobp', 'thirdhops', 'fourthhba', 'fourthhobp', 'fourthhops',
       'fifthhba', 'fifthhobp', 'fifthhops', 'sixthhba', 'sixthhobp',
       'sixthhops', 'seventhhba', 'seventhhobp', 'seventhhops', 'eighth_hba',
       'eighth_hobp', 'eighth_hops', 'ninthhba', 'ninthhobp', 'ninthhops',
       'firstaba', 'firstaobp', 'firstaops', 'secondaba', 'secondaobp',
       'secondaops', 'thirdaba', 'thirdaobp', 'thirdaops', 'fourthaba',
       'fourthaobp', 'fourthaops', 'fifthaba', 'fifthaobp', 'fifthaops',
       'sixthaba', 'sixthaobp', 'sixthaops', 'seventhaba', 'seventhaobp',
       'seventhaops', 'eighthaba', 'eighthaobp', 'eighthaops', 'ninthaba',
       'ninthaobp', 'ninthaops']



df = pd.read_csv('trial5.csv')

x = df[feature_cols]
x[x==np.inf]=np.nan

x.fillna(x.mean(), inplace=True)
col_mask=x.isnull().any(axis=0) 


y = df['h_win']
col_mask=y.isnull().any(axis=0) 
x = x.values
y = y.values

# param_dist = {'n_estimators': np.arange(30, 250),
#               'learning_rate': np.arange(1, 20)/10,
#               'max_depth': np.arange(1, 5),
#               'random_state': np.arange(1, 10)}
clfgtb = GradientBoostingClassifier(random_state = 1, n_estimators = 500, max_depth = 1, learning_rate = 1.2)
# clfgtb = RandomizedSearchCV(clfgtb, param_dist,  cv=15)
cv_score = cross_val_score(clfgtb, x, y, cv = 5)
clfgtb = clfgtb.fit(x, y)
print(cv_score )

df1 = pd.read_csv('2019tester.csv')
df2 = pd.read_csv('julythirdr.csv')
df3 = pd.read_csv('julyfourthr.csv')
df4 = pd.read_csv('julyseventhr.csv')
df5 = pd.read_csv('julysixthr.csv')
df6 = pd.read_csv('julyfifthr.csv')
dfs = [df1, df3, df4, df5, df6]
games = pd.concat(dfs)
games = df1
#new_features = ['a_ML', 'h_ML', 'h_elo_pre', 'a_elo_pre', ]
game_features = ['a_ml', 'h_ml', 'h_elo', 'a_elo', 'era_1', 'pitchesPerPlateAppearance_1', 'whip_1', 'era_0', 'pitchesPerPlateAppearance_0', 'whip_0', 'avg_9', 'obp_9', 'ops_9', 'avg_10', 'obp_10', 'ops_10', 'avg_11', 'obp_11', 'ops_11', 'avg_12', 'obp_12', 'ops_12', 'avg_13', 'obp_13', 'ops_13', 'avg_14', 'obp_14', 'ops_14', 'avg_15', 'obp_15', 'ops_15', 'avg_16', 'obp_16', 'ops_16', 'avg_17', 'obp_17', 'ops_17', 'avg_0', 'obp_0', 'ops_0', 'avg_1', 'obp_1', 'ops_1', 'avg_2', 'obp_2', 'ops_2', 'avg_3', 'obp_3', 'ops_3', 'avg_4', 'obp_4', 'ops_4', 'avg_5', 'obp_5', 'ops_5', 'avg_6', 'obp_6', 'ops_6', 'avg_7', 'obp_7', 'ops_7', 'avg_8', 'obp_8', 'ops_8']
betx = games[new_feature_cols]
h_lines = list(games.h_ML)
a_lines = list(games.a_ML)
# h_team = list(games.home)
# a_team = list(games.away)
winners = list(games.h_win)

allbets = []
betx[betx==np.inf]=np.nan

betx.fillna(betx.mean(), inplace=True)
predictions = clfgtb.predict_proba(betx)
num = len(predictions)
# for i in range(num):

# 	prob1 = preds[i]
# 	line1h = h_lines[i]
# 	line1a = a_lines[i]
# 	hteam = h_team[i]
# 	ateam = a_team[i]
# 	evhome = prob1[1] * yo(line1h) - prob1[0]
# 	evaway = prob1[0] * yo(line1a) - prob1[1]
# 	if evhome > 0:
# 		print('bet on ' + hteam)

# 	if evaway > 0:
# 		print('bet on ' + ateam)
for i in range(num):

    home_winprob = predictions[i][1]
    away_winprob = predictions[i][0]
    winner = winners[i]
    h_line = h_lines[i]
    a_line = a_lines[i]
    evhome = home_winprob * yo(h_line) - away_winprob 
    evaway = away_winprob * yo(a_line) - home_winprob
    if home_winprob > .5 and winner ==1:
        correct = 1

    if away_winprob > .5 and winner ==1:
        correct = 0
    if home_winprob > .5 and winner ==0:
        correct = 0

    if away_winprob > .5 and winner ==0:
        correct = 1

    if winner == 1:
        roi_home = yo(h_line)
        roi_away = -1

    if winner == 0:
        roi_home = -1
        roi_away = yo(a_line)


    if evaway > 0:
        a_bets = [away_winprob, a_line, evaway, roi_away, winner, correct]
        allbets.append(a_bets)
          # print(n)

    if evhome > 0:
          h_bets = [home_winprob, h_line, evhome, roi_home, winner, correct]
          allbets.append(h_bets)

all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner', 'correct'])

print(all_df['roi'].mean())
print(all_df['correct'].mean())
print(all_df['ev'].mean())
print(all_df['winner'].mean())
print(all_df['roi'].sum())
print(all_df.shape)


