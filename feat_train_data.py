import MySQLdb,csv
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
import numpy as np
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
	
def preprocess():
	db = getdb()
	seasons = get_seasons()
	season_arr = []
	season_str = ""
	for s in seasons:
		season_arr.append(s[0])
		season_str += ("'"+s[0]+"',")
	season_str = season_str[0:len(season_str)-1]
	print season_str
	query = "SELECT id from teams"
	cursor = db.cursor()
	lines = cursor.execute(query)
	teams = cursor.fetchall()
	teams_arr = []
	for s in teams:
		teams_arr.append(s[0])
	print teams_arr

	train_dataset = []
	feat_gen = open("feats.csv","w+")
	feat_gen.write("team_id,season,team_wins,team_wscore,team_loss,team_lscore\n")
	for team in teams_arr:
		query = "SELECT season,count(*),sum(wscore) from regular_season_results WHERE season in ("+season_str+") and wteam="+str(team)+" group by season"
		cursor = db.cursor()
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		temp_arr = []
		query = "SELECT season,count(*),sum(lscore) from regular_season_results WHERE season in ("+season_str+") and lteam="+str(team)+" group by season"
		cursor = db.cursor()
		lines = cursor.execute(query)
		loss = cursor.fetchall()
		for i in range(0,len(wins)):
			query = "INSERT INTO summary_play_results(team_id,season,team_wins,team_wscore,team_loss,team_lscore) VALUES("+str(team)+",'"+str(wins[i][0])+"',"+str(wins[i][1])+","+str(wins[i][2])+","+str(loss[i][1])+","+str(loss[i][2])+")"
			cursor = db.cursor()
			lines = cursor.execute(query)
			feat_gen.write(str(team)+","+str(wins[i][0])+","+str(wins[i][1])+","+str(wins[i][2])+","+str(loss[i][1])+","+str(loss[i][2])+"\n")
			tup = (team,wins[i][1],wins[i][2],loss[i][1],loss[i][2],)
			train_dataset.append(tup)
			print tup
		db.commit()
	#print train_dataset
	feat_gen.close()
	db.close()

def preprocess_tour():
	db = getdb()
	seasons = get_seasons()
	season_arr = []
	season_str = ""
	for s in seasons:
		season_arr.append(s[0])
		season_str += ("'"+s[0]+"',")
	season_str = season_str[0:len(season_str)-1]
	print season_str
	query = "SELECT id from teams"
	cursor = db.cursor()
	lines = cursor.execute(query)
	teams = cursor.fetchall()
	teams_arr = []
	for s in teams:
		teams_arr.append(s[0])
	print teams_arr

	train_dataset = []
	feat_gen = open("feats_tourney.csv","w+")
	feat_gen.write("team_id,seed,season,team_wins,team_wscore\n")
	for team in teams_arr:
		query = "SELECT season,count(*),sum(wscore) from regular_season_results WHERE season in ("+season_str+")  and wteam="+str(team)+" group by season"
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
		for i in range(0,len(wins)):
			if wins[i][0] in season_seed:
				seed = season_seed[wins[i][0]]
			else:
				seed = 0
			query = "INSERT INTO summary_play_results(team_id,seed,season,team_wins,team_wscore) VALUES("+str(team)+","+str(seed)+",'"+str(wins[i][0])+"',"+str(wins[i][1])+","+str(wins[i][2])+")"
			cursor = db.cursor()
			lines = cursor.execute(query)
			feat_gen.write(str(team)+","+str(seed)+","+str(wins[i][0])+","+str(wins[i][1])+","+str(wins[i][2])+"\n")
			tup = (team,wins[i][1],wins[i][2],seed,)
			#train_dataset.append(tup)
			print tup
		db.commit()
	#print train_dataset
	feat_gen.close()
	db.close()
	
#pred=>0-loss,1-win
def prepare_tourney_train_data():
	db = getdb()
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	feat_gen = open("train_tourney_set.csv","w+")
	feat_gen.write("pred,season,team1,team2,team1_wins,team1_wscore,team1_seed,team2_wins,team2_wscore,team2_seed,win_score\n")
	seasons = get_seasons()
	train_dataset = []
	for row in csvRdr:
		teams = row[0]
		print teams
		toks = teams.split("_")
		team1 = toks[1]
		team2 = toks[2]
		for s in seasons:
			query = "select team_id,team_wins,team_wscore,seed from summary_play_results where team_id="+str(team1)+" and season='"+s[0]+"'"
			cursor = db.cursor()
			lines = cursor.execute(query)
			rset1 = cursor.fetchall()
			query = "select team_id,team_wins,team_wscore,seed from summary_play_results where team_id="+str(team2)+" and season='"+s[0]+"'"
			cursor = db.cursor()
			lines = cursor.execute(query)
			rset2 = cursor.fetchall()
			tup = ()
			both = 0
			if len(rset1)!=0:
				tup = tup+(team1,rset1[0][1],rset1[0][2],rset1[0][3])
			else:
				tup = tup+(team1,0,0,0)
				both+=1 
			if len(rset2)!=0:
				tup = tup+(team2,rset2[0][1],rset2[0][2],rset2[0][3])
			else:
				tup = tup+(team2,0,0,0)
				both+=1
			if both==2:
				continue
			train_dataset.append(tup)
			if len(tup)!=0:
				#team1_wscore = tup[2]
				#team2_wscore = tup[6]
				team1_wins = tup[1]
				team2_wins = tup[5]
				team1_seed = tup[3]
				team2_seed = tup[7]
				pred = 0
				if team1_seed == team2_seed:
					pred = 0.5
				if team1_seed > team2_seed:
					pred = 1
				win_score = (0.5+(0.03*(team1_seed-team2_seed)))
				feat_gen.write(str(pred)+","+s[0]+","+str(team1)+","+str(team2)+","+str(tup[1])+","+str(tup[2])+","+str(tup[3])+","+str(tup[5])+","+str(tup[6])+","+str(tup[7])+","+str(win_score)+"\n")
	feat_gen.close()
	return train_dataset

#pred=>0-loss,1-win
def prepare_train_data():
	db = getdb()
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	feat_gen = open("train_set.csv","w+")
	feat_gen.write("pred,season,team1,team2,team1_wins,team1_wscore,team1_loss,team1_lscore,team2_wins,team2_wscore,team2_loss,team2_lscore\n")
	seasons = get_seasons()
	train_dataset = []
	for row in csvRdr:
		teams = row[0]
		print teams
		toks = teams.split("_")
		team1 = toks[1]
		team2 = toks[2]
		for s in seasons:
			query = "select team_id,team_wins,team_wscore,team_loss,team_lscore from summary_play_results where team_id in ("+str(team1)+","+str(team2)+") and season='"+s[0]+"'"
			cursor = db.cursor()
			lines = cursor.execute(query)
			rset = cursor.fetchall()
			tup = ()
			for r in rset:
				tup = tup+(r[0],r[1],r[2],r[3],r[4],)
			if len(tup) == 5:
				tup = tup + (0,0,0,0,0,)
			train_dataset.append(tup)
			if len(tup)!=0:
				team1_wscore = tup[2]
				team2_wscore = tup[7]
				pred = 0
				if team1_wscore > team2_wscore:
					pred = 1
				feat_gen.write(str(pred)+","+s[0]+","+str(team1)+","+str(team2)+","+str(tup[1])+","+str(tup[2])+","+str(tup[3])+","+str(tup[4])+","+str(tup[6])+","+str(tup[7])+","+str(tup[8])+","+str(tup[9])+"\n")
	feat_gen.close()
	return train_dataset
		
def train_model():
	df = pd.read_csv("train_tourney_set.csv")
	train_cols =  df.columns[4:]
	#print df["pred"]
	
	logit = sm.Logit(df['pred'], df[train_cols])
	# fit the model
	result = logit.fit()
	print result.summary()
	'''
	sets = np.array(df[train_cols])
	preds =  np.asarray(df['pred'],dtype=np.float32)
	print str(len(sets))+"---"+str(len(preds))
	labels = []
	for p in preds:
		labels.append(int(p))
	svr = SVR()
	result = svr.fit(sets,labels)
	'''
	return result
	

def test_model(result):
	pred_seasons = {'N':13,'O':14,'P':15,'Q':16,'R':17}
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	submission = open("final_sub.csv","w+")
	db = getdb()
	cursor = db.cursor()
	seasons = get_seasons()
	season_str = {}
	for pred_season in pred_seasons:
		sub_seasons = seasons[0:pred_seasons[pred_season]]
		st = ""
		for s in sub_seasons:
			st += ("'"+s[0]+"',")
		season_str[pred_season] = st[0:len(st)-1]
	print season_str
	for row in csvRdr:
		teams = row[0]
		print teams
		toks = teams.split("_")
		season = toks[0]
		team1 = toks[1]
		team2 = toks[2]
		seasons = season_str[season]
		#team1
		query = "SELECT count(*),sum(wscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and wteam="+team1
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		team1_win = wins[0][0]
		#print wins
		team1_winscore = wins[0][1]
		if team1_winscore==None:
			team1_winscore = 0
		query = "SELECT count(*),sum(lscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and lteam="+team1
		lines = cursor.execute(query)
		loss = cursor.fetchall()
		team1_loss = loss[0][0]
		#print wins
		team1_lscore = loss[0][1]
		if team1_lscore==None:
			team1_lscore = 0
		#print team1+"-->"+str(seeds)
		#team2
		query = "SELECT count(*),avg(wscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and wteam="+team2
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		team2_win = wins[0][0]
		team2_winscore = wins[0][1]
		if team2_winscore==None:
			team2_winscore = 0
		query = "SELECT count(*),sum(lscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and lteam="+team2
		lines = cursor.execute(query)
		loss = cursor.fetchall()
		team2_loss = loss[0][0]
		#print wins
		team2_lscore = loss[0][1]
		if team2_lscore==None:
			team2_lscore = 0
		pred = result.predict([[int(team1_win),int(team2_win)]])
		print teams+"---"+str(pred)
		submission.write(teams+","+str(pred[0])+"\n")
	submission.close()

	
def test_tour_model(result):
	pred_seasons = {'N':13,'O':14,'P':15,'Q':16,'R':17}
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	submission = open("final_sub.csv","w+")
	db = getdb()
	cursor = db.cursor()
	seasons = get_seasons()
	season_str = {}
	for pred_season in pred_seasons:
		sub_seasons = seasons[0:pred_seasons[pred_season]]
		st = ""
		for s in sub_seasons:
			st += ("'"+s[0]+"',")
		season_str[pred_season] = st[0:len(st)-1]
	print season_str
	for row in csvRdr:
		teams = row[0]
		print teams
		toks = teams.split("_")
		season = toks[0]
		team1 = toks[1]
		team2 = toks[2]
		seasons = season_str[season]
		#team1
		query = "SELECT count(*),sum(wscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and wteam="+team1
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		team1_win = wins[0][0]
		if team1_win==None:
			team1_win = 0
		#print wins
		team1_winscore = wins[0][1]
		if team1_winscore==None:
			team1_winscore = 0
		query = "SELECT avg(seed) from tourney_seeds WHERE season in ("+seasons+") and team="+team1
		lines = cursor.execute(query)
		seeds = cursor.fetchall()
		team1_seed = seeds[0][0]
		if team1_seed==None:
			team1_seed = 0
		print team1+"-->"+str(team1_seed)
		#team2
		query = "SELECT count(*),sum(wscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and wteam="+team2
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		team2_win = wins[0][0]
		if team2_win==None:
			team2_win = 0
		team2_winscore = wins[0][1]
		if team2_winscore==None:
			team2_winscore = 0
		query = "SELECT avg(seed) from tourney_seeds WHERE season in ("+seasons+") and team="+team2
		lines = cursor.execute(query)
		seeds = cursor.fetchall()
		team2_seed = seeds[0][0]
		if team2_seed==None:
			team2_seed = 0
		print team1+"-->"+str(team2_seed)
		pred = result.predict([[int(team1_win),int(team1_seed),int(team2_win),int(team2_seed),(0.5+(0.03*(team1_seed-team2_seed)))]])
		print teams+"---"+str(pred[0])
		submission.write(teams+","+str(pred[0])+"\n")
	submission.close()
#prediction based on no. of wins and win scores
#preprocess_tour()
dataset = prepare_tourney_train_data()
#result = train_model()
#test_tour_model(result)
'''

'''