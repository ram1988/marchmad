import MySQLdb,csv
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	feat_gen = open("train_tourney_set.csv","w+")
	feat_gen.write("pred,season,team1,team2,team1_winper,team2_winper,team1_ncaa_occ,team2_ncaa_occ,team1_seed,team2_seed\n")
	seasons = get_seasons()
	print seasons
	seasons = seasons[0:13]
	train_dataset = []
	for row in csvRdr:
		teams = row[0]
		print teams
		toks = teams.split("_")
		team1 = toks[1]
		team2 = toks[2]
		for s in seasons:
			query = "select team_id,team_wscore,team_lscore,ncaa_occ,ncaa_semis_finals,win_per,seed from summary_play_results where team_id in ("+str(team1)+","+str(team2)+") and season='"+s[0]+"'"
			cursor = db.cursor()
			lines = cursor.execute(query)
			rset= cursor.fetchall()
			tup = ()
			#if len(rset)!=0:
			if len(rset) == 2:
				print rset
				tup = tup+(team1,rset[0][1],rset[0][2],rset[0][3],rset[0][4],rset[0][5],rset[0][6])
				tup = tup+(team2,rset[1][1],rset[1][2],rset[1][3],rset[1][4],rset[1][5],rset[1][6])
			else:
				continue
			train_dataset.append(tup)
			if len(tup)!=0:
				team1_wscore = tup[1]
				team2_wscore = tup[8]
				team1_lscore = tup[2]
				team2_lscore = tup[9]
				team1_ncaa_occ = tup[3]
				team2_ncaa_occ = tup[10]
				team1_ncaa_sf = tup[4]
				team2_ncaa_sf = tup[11]
				team1_winper = tup[5]
				team2_winper = tup[12]
				team1_seed = tup[6]
				team2_seed = tup[13]
				pred = 0
				wdiff = team1_wscore-team2_wscore
				team_winper = 0
				team_ncaa_occ = 0
				team1_ncaa_sf = 0
				pred = team1_winper
				
				if team1_seed!=0 and team2_seed!=0:
					if team1_seed <= team2_seed:
						pred = pred + (0.02*(team2_seed-team1_seed))
					else:
						pred = pred - (0.02*(team1_seed-team2_seed))
				
				if team1_ncaa_occ!=0 or team2_ncaa_occ!=0:
					if team1_ncaa_occ >= team2_ncaa_occ:
						pred = pred + (0.01*(team1_ncaa_occ-team2_ncaa_occ))
					else:
						pred = pred - (0.01*(team2_ncaa_occ-team1_ncaa_occ))
					
				'''
				if team1_winper>team2_winper and team1_ncaa_occ > team2_ncaa_occ  and team1_seed < team2_seed:
					pred = team1_winper+(0.02*(team2_seed-team1_seed))+(0.01*(team1_ncaa_occ-team2_ncaa_occ))
					team_winper = team1_winper
					team_ncaa_occ = team1_ncaa_occ
					team_ncaa_sf = team1_ncaa_sf
					team_seed = team1_seed
				else:
					pred = (1-team2_winper)+(0.02*(team1_seed-team2_seed))-(0.01*(team2_ncaa_occ-team1_ncaa_occ))
					team_winper = team2_winper
					team_ncaa_occ = team2_ncaa_occ
					team_ncaa_sf = team2_ncaa_sf
					team_seed = team2_seed
				'''
				feat_gen.write(str(pred)+","+s[0]+","+str(team1)+","+str(team2)+","+str(team1_winper)+","+str(team2_winper)+","+str(team1_ncaa_occ)+","+str(team2_ncaa_occ)+","+str(team1_seed)+","+str(team2_seed)+"\n")
				#feat_gen.write(str(pred)+","+s[0]+","+str(team1)+","+str(team2)+","+str(wdiff)+","+str(team_ncaa_occ)+","+str(team_ncaa_sf)+","+str(team_winper)+"\n")
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
	logit = RandomForestClassifier()
	result = logit.fit(sets,labels)
	'''
	return result

def test_model(result):
	pred_seasons = {'N':13,'O':14,'P':15,'Q':16,'R':17}
	csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
	submission = open("final_sub.csv","w+")
	db = getdb()
	cursor = db.cursor()
	for row in csvRdr:
		teams = row[0]
		#print teams
		toks = teams.split("_")
		season = "'"+toks[0]+"'"
		team1 = toks[1]
		team2 = toks[2]
		query = "select team_id,team_wscore,team_lscore,ncaa_occ,ncaa_semis_finals,win_per,seed from summary_play_results where team_id in ("+str(team1)+","+str(team2)+") and season="+season
		cursor = db.cursor()
		lines = cursor.execute(query)
		rset= cursor.fetchall()
		tup = ()
		#if len(rset)!=0:
		if len(rset) == 2:
			print rset
			tup = tup+(team1,rset[0][1],rset[0][2],rset[0][3],rset[0][4],rset[0][5],rset[0][6])
			tup = tup+(team2,rset[1][1],rset[1][2],rset[1][3],rset[1][4],rset[1][5],rset[1][6])
		if len(tup)!=0:
			team1_wscore = tup[1]
			team2_wscore = tup[8]
			team1_lscore = tup[2]
			team2_lscore = tup[9]
			team1_ncaa_occ = tup[3]
			team2_ncaa_occ = tup[10]
			team1_ncaa_sf = tup[4]
			team2_ncaa_sf = tup[11]
			team1_winper = tup[5]
			team2_winper = tup[12]
			team1_seed = tup[6]
			team2_seed = tup[13]
			wdiff = team1_wscore-team2_wscore
			team_winper = 0
			team_ncaa_occ = 0
			team1_ncaa_sf = 0

			if wdiff>0 and team1_winper>team2_winper and team1_ncaa_occ >= team2_ncaa_occ  and team1_seed <= team2_seed:
					team_winper = team1_winper
					team_ncaa_occ = team1_ncaa_occ
					team_ncaa_sf = team1_ncaa_sf
					team_seed = team1_seed
			else:
					team_winper = team2_winper
					team_ncaa_occ = team2_ncaa_occ
					team_ncaa_sf = team2_ncaa_sf
					team_seed = team2_seed
			#pred = result.predict([[team1_winper,team2_winper]])
			#pred = result.predict([[wdiff,team1_ncaa_occ,team2_ncaa_occ,team1_winper,team2_winper]])
			pred = result.predict([[wdiff,team_ncaa_occ,team_winper,team_seed]])
			#print teams+"---"+str(pred)
			submission.write(teams+","+str(round(pred[0],3))+"\n")
			#submission.write(teams+","+str(round(pred[0][0],5))+"\n")
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
	#print season_str
	seasons="'A','B','C','D','E','F','G','H','I','J','K','L','M'"
	for row in csvRdr:
		teams = row[0]
		#print teams
		toks = teams.split("_")
		season = "'"+toks[0]+"'"
		team1 = toks[1]
		team2 = toks[2]
		#seasons = season_str[season]
		#team1
		#query = "SELECT win,wscore,loss,lscore from (SELECT lteam,count(*) as loss,avg(lscore) as lscore from regular_season_results WHERE daynum>135 and season in ("+seasons+")  and lteam="+str(team1)+") a,(SELECT wteam,count(*) as win,avg(wscore) as wscore from regular_season_results WHERE daynum>135 and season in ("+seasons+")  and wteam="+str(team1)+") b where a.lteam=b.wteam"
		query = "SELECT win,wscore,loss,lscore from (SELECT lteam,count(*) as loss,avg(lscore) as lscore from regular_season_results WHERE season in ("+season+")  and lteam="+str(team1)+") a,(SELECT wteam,count(*) as win,avg(wscore) as wscore from regular_season_results WHERE season in ("+season+")  and wteam="+str(team1)+") b where a.lteam=b.wteam"
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		if len(wins)!=0:
			team1_wscore = wins[0][1]
			if team1_wscore==None:
				team1_wscore = 0
			team1_lscore = wins[0][3]
			if team1_lscore==None:
				team1_lscore = 0	
		else:
			team1_wscore = 0
			team1_lscore = 0
		'''
		query = "SELECT avg(seed) from tourney_seeds WHERE season in ("+seasons+") and team="+team1
		lines = cursor.execute(query)
		seeds = cursor.fetchall()
		team1_seed = seeds[0][0]
		if team1_seed==None:
			team1_seed = 0
		'''
		#team2
		#query = "SELECT  win,wscore,loss,lscore from (SELECT lteam,count(*) as loss,avg(lscore) as lscore from regular_season_results WHERE daynum>135 and season in ("+seasons+")  and lteam="+str(team2)+") a,(SELECT wteam,count(*) as win,avg(wscore) as wscore from regular_season_results WHERE daynum>135 and season in ("+seasons+")  and wteam="+str(team2)+") b where a.lteam=b.wteam"
		query = "SELECT  win,wscore,loss,lscore from (SELECT lteam,count(*) as loss,avg(lscore) as lscore from regular_season_results WHERE season in ("+season+")  and lteam="+str(team2)+") a,(SELECT wteam,count(*) as win,avg(wscore) as wscore from regular_season_results WHERE season in ("+season+")  and wteam="+str(team2)+") b where a.lteam=b.wteam"
		lines = cursor.execute(query)
		wins = cursor.fetchall()
		if len(wins)!=0:
			team2_wscore = wins[0][1]
			if team2_wscore==None:
				team2_wscore = 0
			team2_lscore = wins[0][3]
			if team2_lscore==None:
				team2_lscore = 0	
		else:
			team2_wscore = 0
			team2_lscore = 0
		'''
		query = "SELECT avg(seed) from tourney_seeds WHERE season in ("+seasons+") and team="+team2
		lines = cursor.execute(query)
		seeds = cursor.fetchall()
		team2_seed = seeds[0][0]
		if team2_seed==None:
			team2_seed = 0
		'''
		windiff = team1_wscore-team2_wscore
		lossdiff = team1_lscore-team2_lscore
		#print "Team1--->"+str(team1_wscore)+"---"+str(team1_lscore)+"\nTeam2--->"+str(team2_wscore)+"---"+str(team2_lscore)+"\n\n"
		#pred = result.predict([[int(team1_wscore),int(team1_lscore),int(team2_wscore),int(team2_lscore)]])
		pred = result.predict([[int(windiff),int(lossdiff)]])
		print teams+"---"+str(pred)
		submission.write(teams+","+str(round(pred[0],3))+"\n")
		'''
		#pred = result.predict_proba([[int(team1_win),int(team1_seed),int(team2_win),int(team2_seed)]])
		windiff = team1_wscore-team2_wscore
		lossdiff = team1_lscore-team2_lscore
		pred = result.predict_proba([[int(windiff),int(lossdiff)]])
		#pred = result.predict_proba([[int(team1_wscore),int(team1_lscore),int(team2_wscore),int(team2_lscore)]])
		print teams+"---"+str(pred)
		submission.write(teams+","+str(round(pred[0][1],2))+"\n")
		'''
	submission.close()
#prediction based on no. of wins and win scores
#preprocess_tour()
dataset = prepare_tourney_train_data()
#result = train_model()
#test_model(result)
'''

'''