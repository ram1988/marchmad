import MySQLdb,csv
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

#http://blog.yhathq.com/posts/logistic-regression-and-python.html

def getdb():
	return MySQLdb.connect("localhost", "root", "lumos123", "kaggle")
	
def get_seasons():
	db = getdb()
	query = "SELECT season from season"
	cursor = db.cursor()
	lines = cursor.execute(query)
	seasons = cursor.fetchall()
	return seasons

def preprocess_tour():
	db = getdb()
	seasons = get_seasons()
	season_arr = []
	season_str = ""
	'''
	for s in seasons:
		season_arr.append(s[0])
		season_str += ("'"+s[0]+"',")
	season_str = season_str[0:len(season_str)-1]
	print season_str
	'''
	season_str = "'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R'"
	query = "SELECT id from teams"
	cursor = db.cursor()
	lines = cursor.execute(query)
	teams = cursor.fetchall()
	teams_arr = []
	for s in teams:
		teams_arr.append(s[0])
	print teams_arr
	season_wins = {}
	season_arr = season_str.split(",")
	for s in season_arr:
		query = "SELECT wteam,(count(*)*100/matches) as per from regular_season_results a, (select count(*) as matches from regular_season_results where season="+s+") b where season="+s+" group by wteam"
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		team_dominance = {}
		print s
		for w in wins:
			#print w
			team = int(w[0])
			win_per = w[1]
			team_dominance[team] = win_per
		season_wins[s] = team_dominance
	
	train_dataset = []
	feat_gen = open("feats_tourney.csv","w+")
	feat_gen.write("team_id,seed,season,team_wins,team_wscore,team_loss,team_lscore,ncaa_occur,ncaa_semis_finals,win_per\n")
	for team in teams_arr:
		teamid = str(team)
		query = "SELECT a.season, win, wscore,loss,lscore from (SELECT season,count(*) as loss,avg(lscore) as lscore from regular_season_results WHERE season in ("+season_str+")  and lteam="+str(team)+" group by season) a,(SELECT season,count(*) as win,avg(wscore) as wscore from regular_season_results WHERE season in ("+season_str+")  and wteam="+str(team)+" group by season) b where a.season=b.season"
		print query
		cursor = db.cursor()
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		query = "SELECT season,seed from tourney_seeds WHERE season in ("+season_str+") and team="+str(team)
		lines = cursor.execute(query)
		seeds = cursor.fetchall()
		season_seed = {}
		for seed in seeds:
			season = seed[0]
			seed = int(seed[1][1:3])
			season_seed[season] = seed
		print season_seed
		#collecting ncaa info
		ncaa_info = {}
		for seas in season_arr:
			query = "SELECT 1 from tourney_results WHERE season="+seas+" and (wteam="+teamid+" or lteam="+teamid+")" 
			lines = cursor.execute(query)
			ncaa_pres = cursor.fetchall()
			if ncaa_pres:
				ncaa_pres = ncaa_pres[0][0]
				query = "SELECT COUNT(*) from (SELECT wteam,lteam from tourney_results a WHERE a.season="+seas+" order by daynum desc limit 3) a where wteam="+teamid+" or lteam="+teamid
				lines = cursor.execute(query)
				ncaa_semis_finals = cursor.fetchall()[0][0]
			else:
				ncaa_pres = 0
				ncaa_semis_finals = 0
			ncaa_info[seas] = [ncaa_pres,ncaa_semis_finals]
			
		for i in range(0,len(wins)):
			if wins[i][0] in season_seed:
				seed = season_seed[wins[i][0]]
				ncaa_inf = ncaa_info["'"+wins[i][0]+"'"]
			else:
				seed = 0
				ncaa_inf = [0,0]
			ncaa_occ = str(ncaa_inf[0])
			ncaa_semis = str(ncaa_inf[1])
			season = str(wins[i][0])
			win_per = season_wins["'"+season+"'"][int(teamid)]
			query = "INSERT INTO summary_play_results(team_id,seed,season,team_wins,team_wscore,team_loss,team_lscore,ncaa_occ,ncaa_semis_finals,win_per) VALUES("+teamid+","+str(seed)+",'"+season+"',"+str(wins[i][1])+","+str(wins[i][2])+","+str(wins[i][3])+","+str(wins[i][4])+","+str(ncaa_occ)+","+str(ncaa_semis)+","+str(win_per)+")"
			cursor = db.cursor()
			lines = cursor.execute(query)
			print ncaa_inf
			feat_gen.write(str(team)+","+str(seed)+","+str(wins[i][0])+","+str(wins[i][1])+","+str(wins[i][2])+","+str(wins[i][3])+","+str(wins[i][4])+","+str(ncaa_occ)+","+str(ncaa_semis)+","+str(win_per)+"\n")
		db.commit()
	#print train_dataset
	feat_gen.close()
	db.close()
	
#pred=>0-loss,1-win
def prepare_tourney_train_data():
	db = getdb()
	csvRdr = csv.reader(open("data2014/sample_submission.csv", "r"))
	feat_gen = open("train_tourney_set_extra.csv","w+")
	feat_gen.write("pred,season,team1,team2,wdiff,team1_ncaa_occ,team2_ncaa_occ,team1_seed,team2_seed\n")
	
	print "start querying"
	query = "select season,wteam,lteam,wscore,lscore from regular_season_results where season in ('N','O','P','Q','R') order by season"
	cursor = db.cursor()
	lines = cursor.execute(query)
	rset = cursor.fetchall()
	print "get "
	
	for row in rset:
		season = row[0]
		team1 = str(row[1])
		team2 = str(row[2])
		wscore = row[3]
		lscore = row[4]
		
		
		#ncaa_presence
		query = "SELECT count(*) from tourney_results WHERE season='"+season+"' and (wteam="+team1+" or lteam="+team1+")" 
		cursor = db.cursor()
		lines = cursor.execute(query)
		team1_ncaa_occ = cursor.fetchall()[0][0]
		query = "SELECT count(*) from tourney_results WHERE season='"+season+"' and (wteam="+team2+" or lteam="+team2+")" 
		cursor = db.cursor()
		lines = cursor.execute(query)
		team2_ncaa_occ = cursor.fetchall()[0][0]
		
		#seeds
		query = "SELECT seed from tourney_seeds WHERE season='"+season+"' and team="+team1
		lines = cursor.execute(query)
		team1_seed = cursor.fetchall()
		if len(team1_seed)!=0:
			team1_seed = team1_seed[0][0]
			team1_seed = int(team1_seed[1:3])
		else:
			team1_seed = 17
		query = "SELECT seed from tourney_seeds WHERE season='"+season+"' and team="+team2
		lines = cursor.execute(query)
		team2_seed = cursor.fetchall()
		if len(team2_seed)!=0:
			team2_seed = team2_seed[0][0]
			team2_seed = int(team2_seed[1:3])
		else:
			team2_seed = 17
		
		#avg. wscore, lscore
		query = "SELECT wscore,lscore from (SELECT avg(lscore) as lscore from regular_season_results WHERE lteam="+team1+") a,(SELECT avg(wscore) as wscore from regular_season_results WHERE wteam="+team1+") b"
		lines = cursor.execute(query)
		team1_avg = cursor.fetchall()
		team1_avgwscore = team1_avg[0][0]
		team1_avglscore = team1_avg[0][1]
		query = "SELECT wscore,lscore from (SELECT avg(lscore) as lscore from regular_season_results WHERE lteam="+team2+") a,(SELECT avg(wscore) as wscore from regular_season_results WHERE wteam="+team2+") b"
		lines = cursor.execute(query)
		team2_avg = cursor.fetchall()
		team2_avgwscore = team2_avg[0][0]
		team2_avglscore = team2_avg[0][1]
		wdiff = round((team1_avgwscore-team2_avgwscore)/team1_avgwscore,5)
		ldiff = (team1_avglscore-team2_avglscore)/team1_avglscore
		
		print "season"+season+"--->"+team1+"--"+team2+"----seed1"+str(team1_seed)+"---seed2"+str(team2_seed)
		pred = 0.5+wdiff+(0.02*(team2_seed-team1_seed))+(0.01*(team1_ncaa_occ-team2_ncaa_occ))
		if pred > 1:
			pred = 1
		
		feat_gen.write(str(pred)+","+season+","+team1+","+team2+","+str(wdiff)+","+str(team1_ncaa_occ)+","+str(team2_ncaa_occ)+","+str(team1_seed)+","+str(team2_seed)+"\n")
				
	feat_gen.close()
	return train_dataset

	
def logloss(y, yhat):
    #Y and Yhat must be vectors of equal length    
    if len(y) != len(yhat):
        raise UserWarning, 'Y and Yhat are not the same size'
        return
    #We do not predict 0 or 1 as they would make our answers infinitly wrong
    not_allowed = [0,1]
    if len([i for i in yhat if i in not_allowed])>0:
        raise UserWarning, 'You cannot predict 0 or 1'
        return
    from math import log    
    score = -sum(map(lambda y,yhat: y*log(yhat) + (1-y)*log(1-yhat), y,yhat))/len(y)
    return score
	
def train_model():
	df = pd.read_csv("train_tourney_set.csv")
	train_cols =  df.columns[4:]
	#print df["pred"]
	print "training the model"

	logit = sm.Logit(df['pred'], df[train_cols])
	# fit the model
	result = logit.fit()
	print "Cross Fold Validation"
	scores = cross_validation.cross_val_score(logit, df[train_cols], df['pred'], cv=5)
	print result.summary()
	'''
	sets = np.array(df[train_cols])
	preds =  np.asarray(df['pred'],dtype=np.float32)
	print str(len(sets))+"---"+str(len(preds))
	labels = []
	for p in preds:
		labels.append(int(p))
	logit = LogisticRegression()
	result = logit.fit(sets,labels)
	'''
	return result

def cross_validate_model(k):
	df = pd.read_csv("train_tourney_set.csv")
	train_cols =  df.columns[4:]
	#print df["pred"]
	print "training the model"
	tot_len = len(df)
	eq_size = tot_len/k
	print tot_len
	start = 0
	end = eq_size
	test_data = df[start:end]
	train_data = df[end:tot_len]
	print train_data
	for i in range(1,k+1):
		print "Fold--->"+str(i)
			
		logit = sm.Logit(train_data['pred'], train_data[train_cols])
		# fit the model
		result = logit.fit()
		
		start = end
		end = end+eq_size
		
		test_data = df[start:end]
		test_data_pred = []
		actual_pred = []
		for item in test_data.iterrows():
			pred = result.predict([[item[1]["wdiff"],item[1]["team1_ncaa_occ"],item[1]["team2_ncaa_occ"],item[1]["team1_seed"],item[1]["team2_seed"]]])
			test_data_pred.append(pred)
			actual_pred.append(pred)
		score = logloss(actual_pred,test_data_pred)
		print "LogLoss-->"+str(score[0])+"\n\n"
		train_data = [df[0:start] , df[end:tot_len]]
		train_data = pd.concat(train_data)

	
def test_model(result):
	pred_seasons = {'N':13,'O':14,'P':15,'Q':16,'R':17}
	csvRdr = csv.reader(open("data2014/sample_submission.csv", "r"))
	submission = open("final_sub.csv","w+")
	db = getdb()
	cursor = db.cursor()
	
	for row in csvRdr:
		teams = row[0]
		#print teams
		toks = teams.split("_")
		season = toks[0]
		team1 = toks[1]
		team2 = toks[2]
		
		#ncaa_presence
		query = "SELECT count(*) from tourney_results WHERE season='"+season+"' and (wteam="+team1+" or lteam="+team1+")" 
		cursor = db.cursor()
		lines = cursor.execute(query)
		team1_ncaa_occ = cursor.fetchall()[0][0]
		query = "SELECT count(*) from tourney_results WHERE season='"+season+"' and (wteam="+team2+" or lteam="+team2+")" 
		cursor = db.cursor()
		lines = cursor.execute(query)
		team2_ncaa_occ = cursor.fetchall()[0][0]
		
		#seeds
		query = "SELECT seed from tourney_seeds WHERE season='"+season+"' and team="+str(team1)
		lines = cursor.execute(query)
		team1_seed = cursor.fetchall()
		if len(team1_seed)!=0:
			team1_seed = team1_seed[0][0]
			team1_seed = int(team1_seed[1:3])
		else:
			team1_seed = 17
			
		query = "SELECT seed from tourney_seeds WHERE season='"+season+"' and team="+str(team2)
		lines = cursor.execute(query)
		team2_seed = cursor.fetchall()
		if len(team2_seed)!=0:
			team2_seed = team2_seed[0][0]
			team2_seed = int(team2_seed[1:3])
		else:
			team2_seed = 17
		
		#avg. wscore, lscore
		query = "SELECT wscore,lscore from (SELECT avg(lscore) as lscore from regular_season_results WHERE lteam="+str(team1)+") a,(SELECT avg(wscore) as wscore from regular_season_results WHERE wteam="+str(team1)+") b"
		lines = cursor.execute(query)
		team1_avg = cursor.fetchall()
		team1_avgwscore = team1_avg[0][0]
		team1_avglscore = team1_avg[0][1]
		query = "SELECT wscore,lscore from (SELECT avg(lscore) as lscore from regular_season_results WHERE lteam="+str(team2)+") a,(SELECT avg(wscore) as wscore from regular_season_results WHERE wteam="+str(team2)+") b"
		lines = cursor.execute(query)
		team2_avg = cursor.fetchall()
		team2_avgwscore = team2_avg[0][0]
		team2_avglscore = team2_avg[0][1]
		wdiff = round((team1_avgwscore-team2_avgwscore)/team1_avgwscore,5)
		#ldiff = (team1_avglscore-team2_avglscore)/team1_avglscore

		pred = result.predict([[wdiff,team1_ncaa_occ,team2_ncaa_occ,team1_seed,team2_seed]])
		print teams+"---"+str(pred)+"---"+str(wdiff)
		submission.write(teams+","+str(round(pred[0],3))+"\n")
		#submission.write(teams+","+str(round(pred[0][0],5))+"\n")
	submission.close()
#prediction based on no. of wins and win scores
#preprocess_tour()
#dataset = prepare_tourney_train_data()
cross_validate_model(60)
#result = train_model()
#test_model(result)
'''

'''