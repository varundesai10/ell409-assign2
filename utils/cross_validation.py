import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import multiprocessing
from joblib import delayed, Parallel




def calc_metrics(f, train_ind, test_ind, X, Y):
	X_train = X[train_ind]; Y_train = Y[train_ind]
	X_test = X[test_ind]; Y_test = Y[test_ind]
	f.fit(X_train, Y_train)
	
	Y_ = f.predict(X_test)
	conf = np.zeros((2, 2))
	for y, y_ in zip(Y_test, Y_):
		conf[y][y_] += 1
    
	return {'conf':conf, 'acc': (conf[0,0] + conf[1,1])/np.sum(conf), 'prec': conf[1,1]/(conf[1,1] + conf[0,1]), 'rec':conf[1,1]/(conf[1,1] + conf[1,0])}

def cross_validation(arg_dict, X, Y, n_folds=5, random_state=42, grid=False):
	assert(X.shape[0] == Y.shape[0]), 'X and Y have different number of total samples'
	num_cores = min(multiprocessing.cpu_count(), n_folds)
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
	f = SVC(**arg_dict)
	splits = (kf.split(X))
	metrics = Parallel(n_jobs=num_cores)(delayed(calc_metrics)(f, train_ind, test_ind, X, Y) for train_ind, test_ind in splits);
	print(metrics[0])
	if(grid):
		return [{**m, **arg_dict} for m in metrics]
	return metrics

def cross_validation_serial(arg_dict, X, Y, n_folds=5, random_state=42):
	assert(X.shape[0] == Y.shape[0]), 'X and Y have different number of total samples'
	num_cores = min(multiprocessing.cpu_count(), n_folds)
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
	f = SVC(**arg_dict)
	splits = (kf.split(X))
	metrics = [calc_metrics(f, train_ind, test_ind, X, Y) for train_ind, test_ind in splits]
	
	return metrics

