import MySQLdb,csv
import pandas as pd
import statsmodels.api as sm
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

def prepare_train_data():
	db = getdb()
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	feat_gen = open("train_set.csv","w+")
	feat_gen.write("team1,team1_wins,team1_wscore,team1_loss,team1_lscore,team2,team2_wins,team2_wscore,team2_loss,team2_lscore\n")
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
				feat_gen.write(s[0]+","+str(team1)+","+str(team2)+","+str(tup[1])+","+str(tup[2])+","+str(tup[3])+","+str(tup[4])+","+str(tup[6])+","+str(tup[7])+","+str(tup[8])+","+str(tup[9])+"\n")
	feat_gen.close()
	return train_dataset
	
def train_model():
	df = pd.read_csv("train_set.csv")
	train_cols =  df.columns[4:]
	print train_cols
	logit = sm.Logit(df['pred'], df[train_cols])
	# fit the model
	result = logit.fit()
	print result
#preprocess()
#dataset = prepare_train_data()
train_model()
'''

'''