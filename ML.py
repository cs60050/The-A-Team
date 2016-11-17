
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
mainlist=[]
li = [0,1,2,3,4,5,6,7,8,9]


# In[2]:

for m in li:
    input_path ='/home/prithvi/Downloads/MLDATA/' + str(m)
    files_1=(sorted(os.listdir(input_path)))
    files_1 =list(files_1)
    for each in files_1:
        path=input_path+"/"+each
        data = pd.read_csv(path)
        data=data.ix[:,1:4]
        k = int(data.shape[0]/10)
        main=[]
        j=0
        while(j<=8):
            d = {'meanx':data["AccelerometerX"][j*k:(j+1)*k].mean(),'meany':data["AccelerometerY"][j*k:(j+1)*k].mean(),'meanz':data["AccelerometerZ"][j*k:(j+1)*k].mean(),'stdx':data["AccelerometerX"][j*k:(j+1)*k].std(),'stdy':data["AccelerometerY"][j*k:(j+1)*k].mean(),'stdz':data["AccelerometerZ"][j*k:(j+1)*k].std(),'digit':int(each.split("_")[0])}
            main.append(d)
            j=j+1
        if(j*k<=data.shape[0]):
            d = {'meanx':data["AccelerometerX"][j*k:].mean(),'meany':data["AccelerometerY"][j*k:].mean(),'meanz':data["AccelerometerZ"][j*k:].mean(),'stdx':data["AccelerometerX"][j*k:].std(),'stdy':data["AccelerometerY"][j*k:].mean(),'stdz':data["AccelerometerZ"][j*k:].std(),'digit':int(each.split("_")[0])}
            main.append(d)  
        df = pd.DataFrame(main)
        names = df.columns.values 
        l=[]
        for i in names:
            l = l + df[i].tolist()
        mainlist.append(l[9:])


# In[60]:

input_path ='/home/prithvi/Downloads/setof2'
files_1=(sorted(os.listdir(input_path)))
files_1 =list(files_1)
for each in files_1:
    path=input_path+"/"+each
    print path
    data = pd.read_csv(path)
    data=data.ix[:,1:4]
    k = int(data.shape[0]/10)
    main=[]
    j=0
    while(j<=8):
        d = {'meanx':data["AccelerometerX"][j*k:(j+1)*k].mean(),'meany':data["AccelerometerY"][j*k:(j+1)*k].mean(),'meanz':data["AccelerometerZ"][j*k:(j+1)*k].mean(),'stdx':data["AccelerometerX"][j*k:(j+1)*k].std(),'stdy':data["AccelerometerY"][j*k:(j+1)*k].mean(),'stdz':data["AccelerometerZ"][j*k:(j+1)*k].std(),'digit':int(each.split("_")[0])}
        main.append(d)
        j=j+1
    if(j*k<=data.shape[0]):
        d = {'meanx':data["AccelerometerX"][j*k:].mean(),'meany':data["AccelerometerY"][j*k:].mean(),'meanz':data["AccelerometerZ"][j*k:].mean(),'stdx':data["AccelerometerX"][j*k:].std(),'stdy':data["AccelerometerY"][j*k:].mean(),'stdz':data["AccelerometerZ"][j*k:].std(),'digit':int(each.split("_")[0])}
        main.append(d)  
    df = pd.DataFrame(main)
    names = df.columns.values 
    l=[]
    for i in names:
        l = l + df[i].tolist()
    mainlist.append(l[9:])


# In[4]:

dat = pd.DataFrame(mainlist)
#dat.to_csv("sensor_data.csv")


# In[9]:

test = pd.DataFrame(mainlist)


# In[44]:

frames = [train,test]
dat = pd.concat(frames)


# In[57]:

from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, dat.ix[:,1:], dat.ix[:,0], cv=5)
scores.mean()


# In[7]:

from sklearn.model_selection import ShuffleSplit
n_samples = dat.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, dat.ix[:,1:], dat.ix[:,0], cv=cv)                                                     


# In[12]:

clf.fit(dat.ix[:,1:], dat.ix[:,0])


# In[13]:

from sklearn.metrics import confusion_matrix


# In[8]:

from sklearn.model_selection import cross_val_predict
from sklearn import metrics
predicted = cross_val_predict(clf, dat.ix[:,1:], dat.ix[:,0], cv=10)
metrics.accuracy_score(dat.ix[:,0], predicted) 


# In[5]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
train, test = train_test_split(dat, test_size = 0.3)


# In[7]:

train


# In[8]:

test


# In[47]:

from sklearn import svm
from sklearn import metrics
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 


# In[48]:

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(train.ix[:,1:],  train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 
#confusion_matrix(test.ix[:,0], clf.predict(test.ix[:,1:]))


# In[49]:

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train.ix[:,1:],  train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], neigh.predict(test.ix[:,1:])) 


# In[50]:

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
clf.fit(train.ix[:,1:],  train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 


# In[51]:

from sklearn import svm
clf = svm.LinearSVC()
clf.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 


# In[52]:

svm.SVC(decision_function_shape='ovo')
clf.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 


# In[53]:

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 


# In[54]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], gnb.predict(test.ix[:,1:])) 


# In[55]:

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], logreg.predict(test.ix[:,1:])) 
#confusion_matrix(test.ix[:,0], clf.predict(test.ix[:,1:]))


# In[56]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit(train.ix[:,1:], train.ix[:,0])
metrics.accuracy_score(test.ix[:,0], clf.predict(test.ix[:,1:])) 


# In[ ]:



