# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:01:02 2022

@author: v_zamani
"""

import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.cluster import DBSCAN
from matplotlib.lines import Line2D
import math

mat = scipy.io.loadmat(r'D:\Studies\AUT-Master\Data Mining _ Dr Ghatei\7th Project (outlier detection)\pendigits.mat')



key_list = list(mat.keys())

val_list = list(mat.values())


X = pd.DataFrame(val_list[3])
y = pd.DataFrame(val_list[4])





y.columns = ['label']
Data = pd.concat([X,y],axis = 1)

outliers = y.loc[y['label'] == 1,:]
### IS data Imbalanced?
l0 = len(y.loc[y['label'] == 0,:])/len(y)
scp = len(y.loc[y['label'] == 0,:])/len(y.loc[y['label'] == 1,:])
contamination = 1-l0
scale_pos_weight = math.sqrt(contamination) 



######### Visualize Data by PCA

### 2D

pca2 = PCA(n_components=3).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=y['label'], 
                palette='Set1',
                s=100, alpha=0.2).set_title('Real Outliers with Blue color',
                                            fontsize=15)
plt.legend()
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()





############################### Based on assumptions about normal data and outliers:
    
############## Statistical approaches (model-based methods)
##### Parametric
## Mahalaobis distance
## χ2 –statistic




##### Non_Parametric
## IQR




############## Proximity-Base Approaches

##### Nearest-neighbor based algorithms (Distance - Based)
## KNN
knn = KNN(contamination=0.1, metric='manhattan')
y_pred = knn.fit_predict(X)
y_pred = pd.DataFrame(y_pred)

Accuracy = accuracy_score(y, y_pred)
F1_score = f1_score(y, y_pred) 
precision = precision_score(y,y_pred)  
Recall = recall_score(y, y_pred)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')


x=[00.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5]


Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    knn = KNN(contamination=i, metric='manhattan')
    y_pred = pd.DataFrame(knn.fit_predict(X))
    CM = confusion_matrix(y, y_pred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('contamination')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing contamination in knn',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()



### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=y_pred[0], 
                palette='Set1',
                s=100, alpha=0.2).set_title('Detected Outliers by Knn',
                                            fontsize=15)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()




## Average kNN (use the average distance to k nearest neighbors as the outlier score)

avgknn = KNN(method = 'mean',
             contamination=0.1,
             metric='manhattan')


y_pred_avgknn = avgknn.fit_predict(X)
y_pred_avgknn = pd.DataFrame(y_pred_avgknn)

Accuracy = accuracy_score(y, y_pred_avgknn)
F1_score = f1_score(y, y_pred_avgknn) 
precision = precision_score(y,y_pred_avgknn)  
Recall = recall_score(y, y_pred_avgknn)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred_avgknn)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')


x=[00.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5]


Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    knn = KNN(method = 'mean',
             contamination=i,
             metric='manhattan')
    y_pred_avgknn = pd.DataFrame(knn.fit_predict(X))
    CM = confusion_matrix(y, y_pred_avgknn)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred_avgknn)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred_avgknn)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('contamination')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing contamination in AVGknn',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()





### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=y_pred_avgknn[0], 
                palette='Set1',
                s=100, alpha=0.2).set_title('Detected Outliers by AVGKnn',
                                            fontsize=15)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()












##### Density-Based 
##Local Outlier Factor: LOF
lof = LocalOutlierFactor(n_neighbors=10,
                         contamination=0.1,
                         metric='minkowski')

#metrics = cosine, euclidean, manhattan
y_pred_lof = lof.fit_predict(X)
y_pred_lof = pd.DataFrame(y_pred_lof)

mapping = dict(zip([1,-1], range(0, 2)))
y_pred_lof[0] = y_pred_lof[0].map(mapping)


Accuracy = accuracy_score(y, y_pred_lof)
F1_score = f1_score(y, y_pred_lof) 
precision = precision_score(y,y_pred_lof)  
Recall = recall_score(y, y_pred_lof)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred_lof)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')






x=[00.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5]


Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    lof = LocalOutlierFactor(n_neighbors=20,
                         contamination=i,
                         metric='minkowski')
    
    y_pred_lof = pd.DataFrame(lof.fit_predict(X))
    mapping = dict(zip([1,-1], range(0, 2)))
    y_pred_lof[0] = y_pred_lof[0].map(mapping)
    CM = confusion_matrix(y, y_pred_lof)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred_lof)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred_lof)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('contamination')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing contamination in LOF\n n_neighbors = 20',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()






### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=y_pred_lof[0], 
                palette='Set1',
                s=100, alpha=0.2).set_title('Detected Outliers by LOF\n n_neighbors = 10 & contamination = 0.1',
                                            fontsize=15)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()



























## Connectivity Outlier Factor (COF)

cof = COF(contamination=0.1, n_neighbors=20)

y_pred = cof.fit_predict(X)
y_pred = pd.DataFrame(y_pred)



Accuracy = accuracy_score(y, y_pred)
F1_score = f1_score(y, y_pred) 
precision = precision_score(y,y_pred)  
Recall = recall_score(y, y_pred)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')











############## Clustering-Base Approaches
## cluster-based local outlier factor (CBLOF):
    
cblof = CBLOF(contamination=0.1,check_estimator=False, random_state=42)
cblof.fit(X)

y_pred_cblof = cblof.fit_predict(X)
y_pred_cblof = pd.DataFrame(y_pred_cblof)


Accuracy = accuracy_score(y, y_pred_cblof)
F1_score = f1_score(y, y_pred_cblof) 
precision = precision_score(y,y_pred_cblof)  
Recall = recall_score(y, y_pred_cblof)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred_cblof)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')







x=[00.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5]


Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    cblof = CBLOF(contamination=i,
                  check_estimator=False, 
                  random_state=42)

    
    y_pred_cblof = pd.DataFrame(cblof.fit_predict(X))
    CM = confusion_matrix(y, y_pred_cblof)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred_cblof)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred_cblof)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('contamination')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing contamination in CBLOF',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()






### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=y_pred_cblof[0], 
                palette='Set1',
                s=100, alpha=0.2).set_title('Detected Outliers by CBLOF',
                                            fontsize=15)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()
































## DBSCAN

dbscan = DBSCAN(eps = 0.73, min_samples = 100).fit(X)

y_pred_dbscan = pd.DataFrame(dbscan.labels_)
y_pred_dbscan.columns = ['label']


mask = y_pred_dbscan != -1
without = list(y_pred_dbscan[mask['label']]['label'].unique())


y_pred_dbscan['label'] = y_pred_dbscan['label'].replace(without,0)
y_pred_dbscan['label'] = y_pred_dbscan['label'].replace(-1,1)




Accuracy = accuracy_score(y, y_pred_dbscan)
F1_score = f1_score(y, y_pred_dbscan) 
precision = precision_score(y,y_pred_dbscan)  
Recall = recall_score(y, y_pred_dbscan)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred_dbscan)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')











x = np.arange (0.1, 1, 0.1)

Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    dbscan = DBSCAN(eps = i,
                    min_samples = 100).fit(X)

    y_pred_dbscan = pd.DataFrame(dbscan.labels_)
    y_pred_dbscan.columns = ['label']


    mask = y_pred_dbscan != -1
    without = list(y_pred_dbscan[mask['label']]['label'].unique())
    
    
    
    
    y_pred_dbscan['label'] = y_pred_dbscan['label'].replace(without,0)
    y_pred_dbscan['label'] = y_pred_dbscan['label'].replace(-1,1)
        
        
    
    
    CM = confusion_matrix(y, y_pred_dbscan)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred_dbscan)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred_dbscan)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('eps')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing eps in DBSCAN & min_samples = 50',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()






### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue = y_pred_dbscan['label'], 
                palette = 'Set1',
                s=100,
                alpha=0.2)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.title('Detected Outliers by DBSCAN\n eps = 0.73 & min_samples = 100')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()





















######################################  Outlier Ensembles


########### Feature Bagging

FBcblof = FeatureBagging(CBLOF(contamination=0.1,
                          check_estimator=False,
                          random_state=42))


y_pred_fbcblof = FBcblof.fit_predict(X)
y_pred_fbcblof= pd.DataFrame(y_pred_fbcblof)


Accuracy = accuracy_score(y, y_pred_fbcblof)
F1_score = f1_score(y, y_pred_fbcblof) 
precision = precision_score(y,y_pred_fbcblof)  
Recall = recall_score(y, y_pred_fbcblof)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred_fbcblof)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')






### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue = y_pred_fbcblof[0], 
                palette = 'Set1',
                s=100,
                alpha=0.2)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.title('Detected Outliers by FB on CBLOF\n eps = 0.73 & min_samples = 100')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()
















########### IsolationForest
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


print('After OverSampling, the shape of X: {}'.format(X.shape))
print('After OverSampling, the shape of y: {} \n'.format(y.shape))
  
print("After OverSampling, counts of label '1': {}".format(len(y.loc[y['label'] == 0,:])))
print("After OverSampling, counts of label '0': {}".format(len(y.loc[y['label'] == 1,:])))






iso = IsolationForest(contamination=0.1)
y_pred_iso = iso.fit_predict(X)
y_pred_iso = pd.DataFrame(y_pred_iso)

mapping = dict(zip([1,-1], range(0, 2)))
y_pred_iso[0] = y_pred_iso[0].map(mapping)


Accuracy = accuracy_score(y, y_pred_iso)
F1_score = f1_score(y, y_pred_iso) 
precision = precision_score(y,y_pred_iso)  
Recall = recall_score(y, y_pred_iso)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 

CM = confusion_matrix(y, y_pred_iso)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')






### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue = y_pred_iso[0], 
                palette = 'Set1',
                s=100,
                alpha=0.2)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.title('Detected Outliers by IsolationForest\n contamination = 0.1')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()
















########### EllipticEnvelope
ee = EllipticEnvelope(contamination=0.3)
y_pred_ee = ee.fit_predict(X)


y_pred_ee = pd.DataFrame(y_pred_ee)

mapping = dict(zip([1,-1], range(0, 2)))
y_pred_ee[0] = y_pred_ee[0].map(mapping)


Accuracy = accuracy_score(y, y_pred_ee)
F1_score = f1_score(y, y_pred_ee) 
precision = precision_score(y,y_pred_ee)  
Recall = recall_score(y, y_pred_ee)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 

CM = confusion_matrix(y, y_pred_ee)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')






x=[00.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5]


Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    
    ee = EllipticEnvelope(contamination=i)
    y_pred_ee = pd.DataFrame(ee.fit_predict(X))
    


    mapping = dict(zip([1,-1], range(0, 2)))
    y_pred_ee[0] = y_pred_ee[0].map(mapping)

    
    
    
    CM = confusion_matrix(y, y_pred_ee)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred_ee)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred_ee)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('contamination')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing contamination in\n EllipticEnvelope',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()




### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue = y_pred_ee[0], 
                palette = 'Set1',
                s=100,
                alpha=0.2)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.title('Detected Outliers by EllipticEnvelope\n contamination = 0.3')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()










## ABOD (Angle-Based Outlier Detection)


abod = ABOD(contamination=0.42)

y_pred_abod = abod.fit_predict(X)
y_pred_abod = pd.DataFrame(y_pred_abod)

Accuracy = accuracy_score(y, y_pred_abod)
F1_score = f1_score(y, y_pred_abod) 
precision = precision_score(y,y_pred_abod)  
Recall = recall_score(y, y_pred_abod)  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 


CM = confusion_matrix(y, y_pred_abod)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')





x=[00.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5]


Recall = []
False_Alarm_Rate = []
Accuracy = []
for i in x:
    
    abod = ABOD(contamination=i)

    y_pred_abod = abod.fit_predict(X)
    y_pred_abod = pd.DataFrame(y_pred_abod)
    
        
    
    
    CM = confusion_matrix(y, y_pred_abod)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Recall.append(recall_score(y, y_pred_abod)*100)
    False_Alarm_Rate.append((FP/(TN+FP))*100)
    Accuracy.append(accuracy_score(y, y_pred_abod)*100)

plt.plot(x,Recall,label = "Recall")
plt.plot(x,Accuracy,label = "Accuracy")
plt.plot(x,False_Alarm_Rate,label = "False_Alarm_Rate")
plt.xlabel('contamination')
plt.ylabel('measures')
plt.title('Validation Measures Trend by increasing contamination in\n ABOD',
          pad = '16',
          fontweight=400,
          fontsize=15,
          fontstyle='italic')
plt.legend()








### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue = y_pred_abod[0], 
                palette = 'Set1',
                s=100,
                alpha=0.2)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.title('Detected Outliers by ABOD\n contamination = 0.42')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()







































############################### Based on whether user-labeled examples of outliers can be obtained: 


###### Classification Approaches (recall is more important than accuracy)


##### split data into train and validation  set and test data
train, validate, test = \
              np.split(Data.sample(frac=1, random_state=42), 
                       [int(.6*len(Data)), int(.8*len(Data))])


##### split data into X and Y

y_train = train['label']
X_train = train.drop('label', axis=1)

y_validate = validate['label']
X_validate = validate.drop('label', axis=1)


y_test = test['label']
X_test = test.drop('label', axis=1)


#### fit a model on training data



model = XGBClassifier(objective = 'binary:logistic' ,
                      use_label_encoder=False,
                      learning_rate = 0.5,
                      scale_pos_weight = (scp))
model.fit(X_train, y_train,
          eval_set=[(X_validate,y_validate)],
          early_stopping_rounds=5)


# make prediction with XGBOOST
##############################
predictions = pd.DataFrame(model.predict(X_validate))
predictions.columns = ['Predicted Values']


Accuracy = accuracy_score(y_validate, predictions)
F1_score = f1_score(y_validate, predictions,average='macro') 
precision = precision_score(y_validate, predictions,average='macro')  
Recall = recall_score(y_validate, predictions,average='macro')  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 
    
    
    
    
    
CM = confusion_matrix(y_validate, predictions)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')

    
    

    
    
    
    
    
    
    
    
    
    

    
predictions = pd.DataFrame(model.predict(X_test))
predictions.columns = ['Predicted Values']


Accuracy = accuracy_score(y_test, predictions)
F1_score = f1_score(y_test, predictions,average='macro') 
precision = precision_score(y_test, predictions,average='macro')  
Recall = recall_score(y_test, predictions,average='macro')  

measure = [Accuracy, F1_score,precision,Recall]
measure_name = ['Accuracy','F1_Score','precision','Recall']

for i in range(0,len(measure)):
    print(measure_name[i] + ":")
    print("{:.3f}".format(measure[i]*100), '%') 
    
    
    
    
    
CM = confusion_matrix(y_test, predictions)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

False_Alarm_Rate = (FP/(TN+FP))*100
print("False_Alarm_Rate is : ")
print("{:.3f}".format(False_Alarm_Rate),'%')

    

    
    
    
    
    
    
    
    
y_pred_classification = pd.DataFrame(model.predict(X))
y_pred_classification.columns = ['Predicted Values']

    
    
    
### Visualize
pca2 = PCA(n_components=2).fit(X)
pca2d = pca2.transform(X)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue = y_pred_classification['Predicted Values'], 
                palette = 'Set1',
                s=100,
                alpha=0.2)
                                            
                                            
                                            
custom = [Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

                                        
plt.legend(custom,["inlier", "outlier"])
plt.title('Detected Outliers by XGBoost Classifier')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    