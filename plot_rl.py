import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


legends = ['ours', 'baseline', '', '', '']

# add any folder directories here!
log_list = [
			pd.read_csv("trained_models/rl/con40/public_ours_rl/progress.csv"),
			pd.read_csv("trained_models/rl/con40/public_morton_rl/progress.csv"),
	]


logDicts = {}
for i in range(len(log_list)):
	logDicts[i] = log_list[i]

graphDicts={0:'eprewmean', 1:'loss/value_loss'}

legendList=[]
# summarize history for accuracy

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0
	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.plot(logDicts[key]['misc/total_timesteps'],logDicts[key][graphDicts[i]])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='lower right')
	legendList=[]

plt.show()


