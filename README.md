# Outlier-Detection
## In this project, an attempt has been made to compare various methods of detecting outliers in data, including:
* Proximity-Based Approach, 
* Clustering-Base Approaches, 
* Outlier Ensembles,
* and Angle - Based – OutlierDetection (ABOD).

For this purpose, the [pendigits](pendigits.mat) dataset was used.

## First Step
### Reading Data in Python

```python
mat = scipy.io.loadmat(r'D:\Studies\pendigits.mat')


key_list = list(mat.keys())
val_list = list(mat.values())


X = pd.DataFrame(val_list[3])
y = pd.DataFrame(val_list[4])
```

