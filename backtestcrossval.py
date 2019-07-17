import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from itertools import combinations
import itertools
import matplotlib.pyplot as plt
# import bens_helpers as h

def yo(odd):
    # to find the adjusted odds multiplier 
    # returns float
    if odd == 0:
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)





#df = df.apply(pd.to_numeric, errors='coerce')




feature_cols = ['a_ML', 'h_ML', 'elo_h_pre',
       'elo_a_pre', 'a_wins', 'a_losses', 'h_wins', 'h_losses', 'pitcherhera', 'pitcherhpperi', 'pitcherhwhip',
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

#df = df[cols]
first10 = pd.read_csv('no_mls.csv')
last10 = pd.read_csv('hasmls.csv')


chunked = {key: val for key, val in last10.groupby(by='year')}
chunked = pd.DataFrame.from_dict(chunked, orient='index')
chunked = list(chunked.iloc[:, 0])

allframes = chunked
rois = []
dfs= []
run1 = [pd.concat([first10, pd.concat(chunked[0:7])]), chunked[8]]
run2 = [pd.concat([first10, pd.concat(chunked[0:6]), chunked[8]]), chunked[7]]
run3 = [pd.concat([first10, pd.concat(chunked[0:5]), pd.concat(chunked[7:])]), chunked[6]]
run4 = [pd.concat([first10, pd.concat(chunked[0:4]), pd.concat(chunked[6:])]), chunked[5]]
run5 = [pd.concat([first10, pd.concat(chunked[0:3]), pd.concat(chunked[5:])]), chunked[4]]
run6 = [pd.concat([first10, pd.concat(chunked[0:2]), pd.concat(chunked[4:])]), chunked[3]]
run7 = [pd.concat([first10, pd.concat(chunked[0:1]), pd.concat(chunked[3:])]), chunked[2]]
run8 = [pd.concat([first10, chunked[0], pd.concat(chunked[2:])]), chunked[1]]
options = [run1, run2, run3, run4, run5, run6, run7, run8]
for option in options:
    train_ml = option[0]
    testdf = option[1]
    x = train_ml[feature_cols]
    x[x==np.inf]=np.nan

    x.fillna(x.mean(), inplace=True)
    col_mask=x.isnull().any(axis=0) 


    y = train_ml['h_win']
    col_mask=y.isnull().any(axis=0) 
    x = x.values
    y = y.values
    #{'random_state': 5, 'n_estimators': 249, 'max_depth': 1, 'learning_rate': 1.2}
    clfgtb = GradientBoostingClassifier(random_state = 5, n_estimators = 249, max_depth = 1, learning_rate = 1.2)
    #clfgtb = RandomizedSearchCV(clfgtb, param_dist,  cv=100)



    print(x.shape)
    cv_score = cross_val_score(clfgtb, x, y, cv = 10)
    clfgtb = clfgtb.fit(x, y)
    print(cv_score )

    x_test = testdf[feature_cols]
    x_test[x_test==np.inf]=np.nan

    x_test.fillna(x_test.mean(), inplace=True)
    col_mask=x_test.isnull().any(axis=0) 
    y_test = testdf['h_win']
    col_mask=y_test.isnull().any(axis=0) 
    print(clfgtb.score(x_test, y_test))

    predictions = clfgtb.predict_proba(x_test)
    h_lines = testdf['h_ML']
    winners = testdf['h_win']
    winners = list(winners)

    h_lines = list(h_lines)
    #print(h_lines[0])

    a_lines = testdf['a_ML']
    a_lines = list(a_lines)
    h = len(predictions)
    abets = []
    hbets = []
    allbets = []
    n= 0
    for i in range(h):

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


        if evaway > 0 and a_line > -120:
            a_bets = [away_winprob, a_line, evaway, roi_away, winner, correct]
            abets.append(a_bets)
            allbets.append(a_bets)
            n+= roi_away
            # print(n)

        if evhome > 0 and h_line > -120:
            h_bets = [home_winprob, h_line, evhome, roi_home, winner, correct]
            hbets.append(h_bets)
            allbets.append(h_bets)
            n+=roi_home
            # print(n)
        
        
    all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner', 'correct'])
    home_df = pd.DataFrame(hbets, columns = ['home_winprob', 'h_line', 'evhome', 'roi_home', 'winner', 'correct'])
    away_df = pd.DataFrame(abets, columns = ['away_winprob', 'a_line', 'evaway', 'roi_away', 'winner', 'correct'])
    total_roi = all_df['roi'].sum() 
    # print(total_roi)
    # print(all_df)
    rois.append(total_roi)
    home_roi = home_df['roi_home'].sum() 
    # print(total_roi)

    away_roi = away_df['roi_away'].sum() 
    print(rois)
    print('hello martini')
    print(all_df['roi'].mean())
    print(all_df['winprob'].mean())
    print(all_df['ev'].mean())
    print(all_df['correct'].mean())
    print(all_df['line'].mean())
    # below down is graphing to see lines correlated with roi
    # dfs.append(all_df)
    # df = pd.concat(dfs)
    # lines = list(df['line'])
    # groups = []
    # for value in lines:
    #     if value <= -400:
    #         group = 1
    #     if -400<value<=-300:
    #         group = 2
    #     if -300< value <= -250:
    #         group = 3
    #     if -250 < value <= -220:
    #         group = 4
    #     if -220< value <= -200:
    #         group = 5
    #     if -200< value <= -180:
    #         group = 6
    #     if -180< value <= -160:
    #         group = 7
    #     if -160< value <= -130:
    #         group = 8
    #     if -130< value <= 100:
    #         group = 9
    #     if 100< value <= 125:
    #         group = 10
    #     if 125< value <= 140:
    #         group = 11

    #     if 140< value <= 160:
    #         group = 12

    #     if 160< value <= 190:
    #         group = 13
    #     if 190< value <= 220:
    #         group = 14
    #     if 220< value <= 250:
    #         group = 15
    #     if 250< value <= 300:
    #         group = 16
    #     if 300< value:
    #         group = 17
    #     groups.append(group)
    # df['groups'] = groups
    # x = []
    # y = []
    # for line, df_line in df.groupby('groups'):
    #     x.append(list(df_line['line'])[0])
    #     y.append(df_line['roi'].mean())
    # plt.plot(x, y, linewidth=2)
    # #make dfs one df
    # plt.show()
print(rois)
    # print(total_roi)
# perm = combinations(allframes, 8)
# for i in list(perm):
#     print(i)
# for i in range(7):
#     testdf = allframes[i]
#     frames = allframes
#     firstset = frames[0:i]
#     secondset = frames[i+1:]
#     frames = [firstset, secondset]
#     frames = [item for sublist in frames for item in sublist]
#     print(len(frames))
#     print('hello')
#     train_ml = pd.concat(frames)
#     train_ml = pd.concat([train_ml, first10])
#     train_ml = train_ml.sort_values(by='Date')




#     x = train_ml[feature_cols]
#     x[x==np.inf]=np.nan

#     x.fillna(x.mean(), inplace=True)
#     col_mask=x.isnull().any(axis=0) 


#     y = train_ml['h_win']
#     col_mask=y.isnull().any(axis=0) 
#     x = x.values
#     y = y.values

#     clfgtb = GradientBoostingClassifier(random_state = 0, n_estimators = 43, max_depth = 1, learning_rate = 1)
#     #clfgtb = RandomizedSearchCV(clfgtb, param_dist,  cv=100)



#     print(x.shape)
#     cv_score = cross_val_score(clfgtb, x, y, cv = 10)
#     clfgtb = clfgtb.fit(x, y)
#     print(cv_score )

#     x_test = testdf[feature_cols]
#     x_test[x_test==np.inf]=np.nan

#     x_test.fillna(x_test.mean(), inplace=True)
#     col_mask=x_test.isnull().any(axis=0) 
#     print(col_mask)
#     y_test = testdf['h_win']
#     col_mask=y_test.isnull().any(axis=0) 
#     print(col_mask)
#     print(clfgtb.score(x_test, y_test))



#     predictions = clfgtb.predict_proba(x_test)
#     h_lines = testdf['h_ML']
#     winners = testdf['h_win']
#     winners = list(winners)

#     h_lines = list(h_lines)
#     #print(h_lines[0])

#     a_lines = testdf['a_ML']
#     a_lines = list(a_lines)
#     h = len(predictions)
#     abets = []
#     hbets = []
#     allbets = []
#     n= 0
#     for i in range(h):

#         home_winprob = predictions[i][1]
#         away_winprob = predictions[i][0]
#         winner = winners[i]
#         h_line = h_lines[i]
#         a_line = a_lines[i]
#         evhome = home_winprob * yo(h_line) - away_winprob 
#         evaway = away_winprob * yo(a_line) - home_winprob

#         if winner == 1:
#             roi_home = yo(h_line)
#             roi_away = -1

#         if winner == 0:
#             roi_home = -1
#             roi_away = yo(a_line)


#         if evaway > 0:
#             a_bets = [away_winprob, a_line, evaway, roi_away, winner]
#             abets.append(a_bets)
#             allbets.append(a_bets)
#             n+= roi_away
#             #print(n)

#         if evhome > 0:
#             h_bets = [home_winprob, h_line, evhome, roi_home, winner]
#             hbets.append(h_bets)
#             allbets.append(h_bets)
#             n+=roi_home
#             #print(n)
        
        
#     all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
#     home_df = pd.DataFrame(hbets, columns = ['home_winprob', 'h_line', 'evhome', 'roi_home', 'winner'])
#     away_df = pd.DataFrame(abets, columns = ['away_winprob', 'a_line', 'evaway', 'roi_away', 'winner'])

#     total_roi = all_df['roi'].sum() 
#     # print(total_roi)
#     # print(all_df)
#     rois.append(total_roi)
#     home_roi = home_df['roi_home'].sum() 
#     # print(total_roi)

#     away_roi = away_df['roi_away'].sum() 
#     # print(total_roi)

# print(rois)

# model = tf.keras.Sequential([
#     #keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(2, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x, y, epochs=100, batch_size=12)



# train_years = ['2010', '2011', '2013', '2015', '2017']
# test_years = ['2018']
# sport = 'mlb'
# train_fns = [year + 'mlbodds.csv' for year in train_years]
# test_fns = [year + 'mlbodds.csv' for year in test_years]

# def clean(fn):
#     df = pd.read_csv(fn)
#     df = df.dropna()erkfjmfgdnhfdhdfkjk
#     df = df.drop_duplicates()
#     return df  
# def dfs(fns):
#     dfs = []
#     for fn in fns:
#         tmp_df = clean(fn)
#         dfs.append(tmp_df)
#     df = pd.concat(dfs)
#     return df 


# train_df = dfs(train_fns)
# test_df = dfs(test_fns)

# print(train_df.shape)
# print('above is shape')
# feature_cols = ['h_final8rol', 'h_finalfullrol', 'Final8rol', 'Finalfullrol']#, 'h_finala8rol', 'h_finalafullrol', 'Finala8rol', 'Finalafullrol', 'v_wina8rol', 'v_winafullrol', 'h_win8rol', 'h_win8rol', 'awaypast10runscored', 'awaypast10runsallowed', 'awaypast10rec', 'awaypastrunscored', 'awaypastrunsallowed', 'awaypastrec', 'runscored10rol', 'runsallowed10rol', 'result10rol', 'runscoredfullrol', 'runsallowedfullrol', 'resultfullrol']
# x = train_df[feature_cols]
# y = train_df['h_win']

# x = x.values
# y = y.values

# model = tf.keras.Sequential([
#     #keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(2, activation=tf.nn.softmax)
# ])


# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# model.fit(x, y, epochs=100, batch_size=32,)
# test_x = test_df[feature_cols]
# test_y = test_df['h_win']

# test_x = test_x.values
# test_y = test_y.values
# test_loss, test_acc = model.evaluate(test_x, test_y)

# print('Test accuracy:', test_acc)
# test = [[.6, .8]]
# test = np.asarray(test)
# prediction = model.predict(test)
# #prediction = np.argmax(prediction)
# print(prediction)
# prediction = np.argmax(prediction)
# print(prediction)


