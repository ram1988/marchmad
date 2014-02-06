import MySQLdb,csv

pred_seasons = {'N':13,'O':14,'P':15,'Q':16,'R':17}
csvRdr = csv.reader(open("data/sample_submission.csv", "r"))
db = MySQLdb.connect("localhost", "root", "lumos123", "kaggle")
query = "SELECT season from season"
cursor = db.cursor()
lines = cursor.execute(query)
seasons = cursor.fetchall()
season_str = {}
for pred_season in pred_seasons:
	sub_seasons = seasons[0:pred_seasons[pred_season]]
	st = ""
	for s in sub_seasons:
		st += ("'"+s[0]+"',")
	season_str[pred_season] = st[0:len(st)-1]
print season_str

feat_gen = open("feats.csv","w+")
feat_gen.write("team_id,team1_wins,team2_wins,team1_avgscore,team2_avgscore,team1_avgseed,team2_avgseed,pred\n")
for row in csvRdr:
	teams = row[0]
	print teams
	toks = teams.split("_")
	season = toks[0]
	team1 = toks[1]
	team2 = toks[2]
	seasons = season_str[season]
	#team1
	query = "SELECT count(*),avg(wscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and wteam="+team1
	lines = cursor.execute(query)
	wins = cursor.fetchall()
	team1_win = wins[0][0]
	#print wins
	team1_avgscore = wins[0][1]
	if team1_avgscore==None:
		team1_avgscore = 0
	query = "SELECT seed from tourney_seeds WHERE season in ("+seasons+") and team="+team1
	lines = cursor.execute(query)
	seeds = cursor.fetchall()
	#print team1+"-->"+str(seeds)
	avg_seed_team1 = 0
	for seed in seeds:
		seed = int(seed[0][1:3])
		avg_seed_team1 += seed
	if len(seeds)!=0:
		avg_seed_team1 = avg_seed_team1/len(seeds)
	else:
		avg_seed_team1 = 0
	#print avg_seed_team1
	#team2
	query = "SELECT count(*),avg(wscore) from tourney_results WHERE daynum>135 and season in ("+seasons+") and wteam="+team2
	lines = cursor.execute(query)
	wins = cursor.fetchall()
	team2_win = wins[0][0]
	team2_avgscore = wins[0][1]
	if team2_avgscore==None:
		team2_avgscore = 0
	query = "SELECT seed from tourney_seeds WHERE season in ("+seasons+") and team="+team2
	lines = cursor.execute(query)
	seeds = cursor.fetchall()
	avg_seed_team2 = 0
	for seed in seeds:
		seed = int(seed[0][1:3])
		avg_seed_team2 += seed
	if len(seeds)!=0:
		avg_seed_team2 = avg_seed_team2/len(seeds)
	else:
		avg_seed_team2 = 0
	feat_gen.write(teams+","+str(team1_win)+","+str(team2_win)+","+str(team1_avgscore)+","+str(team2_avgscore)+","+str(avg_seed_team1)+","+str(avg_seed_team2)+",0\n")
feat_gen.close()
db.close()	

