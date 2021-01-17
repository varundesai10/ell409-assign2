import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils.smo import SVM 
import time

data = pd.read_csv('health_data.csv')
X = data.to_numpy()[:, :3]
y = data.to_numpy()[:, 3]

mySVM = SVM()
t1 = time.time()
mySVM.fit(X, y)
t2 = time.time()

print(f'Time elapsed = {t2 -t1}s')