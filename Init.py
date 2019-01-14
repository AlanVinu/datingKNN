import os
os.chdir('C:/Users/Alan/Documents/Python/Machine Learning in Action')

import kNN
group, labels = kNN.createDataSet()
kNN.classify0([0,0], group, labels, 3)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], label = 'Data')
from numpy import *
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
           15.0*array(datingLabels), 15.0*array(datingLabels))
ax.set_title('Scatterplot')
ax.set_xlabel('Percentage of Time Spent Playing Games')
ax.set_ylabel('Liters of Ice Cream Consumed per Week')
ax.legend(loc = 'best')
plt.show()

# To reload file : -
#import importlib
#importlib.reload(kNN)
