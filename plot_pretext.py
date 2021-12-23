import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


legends = ['ours', 'baseline', '', '', '', '']


# add any folder directories here!
log_list = [
pd.read_csv("trained_models/pretext/public_ours/progress.csv"),
pd.read_csv("trained_models/pretext/public_morton/progress.csv"),
			]


logDicts = {}
for i in range(len(log_list)):
	logDicts[i] = log_list[i]

# graphDicts={0:'loss'}
graphDicts={0:'loss', 1:'act_loss', 2: 'kl_loss'}

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
			plt.plot(logDicts[key]['epoch'],logDicts[key][graphDicts[i]])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('number of epochs')
	bottom, top = plt.ylim()  # return the current ylim
	plt.ylim((0, top))
	plt.legend(legendList, loc='upper right')
	legendList=[]



plt.show()


