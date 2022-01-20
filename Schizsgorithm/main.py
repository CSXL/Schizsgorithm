import os
import mne
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,GridSearchCV
from glob import glob
from scipy import stats

all_data = os.path.join('datafiles/')
def read_file(file):
  data = mne.io.read_raw_edf(file,preload=True)
  data.set_eeg_reference()
  data.filter(l_freq=0.5,h_freq=60)
  epoch = mne.make_fixed_length_epochs(data,duration=5,overlap=1)
  array = epoch.get_data()
  return array

#data loading
healthy_people_data = [read_file('datafiles/'+file) for file in os.listdir(all_data) if file[0]=='h']
schiziophernic_people_data = [read_file('datafiles/'+file) for file in os.listdir(all_data) if file[0]=='s']
#label loading
healthy_people_labels = [len(i)*[0]for i in healthy_people_data]
schiziophernic_people_labels = [[1]*len(i)for i in schiziophernic_people_data]


data = healthy_people_data+schiziophernic_people_data
labels = healthy_people_labels+schiziophernic_people_labels
group_list = [[i]*len(j) for i,j in enumerate(data)]


data = np.vstack(data)
labels = np.hstack(labels)
group_list = np.hstack(group_list)


#functions for feature concatenation
def mean(x):
  return np.mean(x,axis=-1)
def std(x):
  return np.std(x,axis=-1)
def ptp(x):
  return np.ptp(x,axis=-1)
def var(x):
  return np.var(x,axis=-1)
def minim(x):
  return np.min(x,axis=-1)
def maxim(x):
  return np.max(x,axis=-1)
def argmin(x):
  return np.argmin(x,axis=-1)
def argmax(x):
  return np.argmax(x,axis=-1)
def rms(x):
  return np.sqrt(np.mean(x**2,axis=-1))
def abs_diff_sigal(x):
  return np.sum(np.abs(np.diff(x,axis=-1)),axis=-1)
def skewness(x):
  return stats.skew(x,axis=-1)
def kurtosis(x):
  return stats.kurtosis(x,axis=-1)
def concatenation_feature_data(x):
  return np.concatenate((mean(x),std(x),ptp(x),var(x),minim(x),maxim(x),argmin(x),
                         argmax(x),rms(x),abs_diff_sigal(x),skewness(x),kurtosis(x)),axis=-1)

features = []
for d in data:
  features.append(concatenation_feature_data(d))


clf = LogisticRegression()
gkf = GroupKFold(5)
pipeline = Pipeline([('scalar',StandardScaler()),('clf',clf)])
param_grid={'clf__C':[0.1,0.6,2,3,4,7,34]}
gscv =GridSearchCV(pipeline,param_grid,cv=gkf,n_jobs=2)
gscv.fit(features,labels,groups=group_list)
print(gscv.best_score_*100)