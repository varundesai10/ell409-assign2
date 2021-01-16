import numpy as np
from sklearn.svm import SVC
import multiprocessing
from joblib import delayed, Parallel
import time
from . import cross_validation
from itertools import product

def calc_avg_metrics(metrics):
    keys = ['acc', 'prec', 'rec']
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = sum([ m[key] for m in metrics])/len(metrics)
    return avg_metrics 

def grid_search(arg_dict, X, Y, gamma_list, c_list, n_folds=5, random_state=42, metric='acc', n_procs=3):
	num_cores = min(multiprocessing.cpu_count(), n_procs)
	iteration = product(gamma_list, c_list)
	start = time.time()
	print('Performing grid search over gamma, c...', end=' ')
	metrics = Parallel(n_jobs=num_cores)(delayed(cross_validation.cross_validation)(arg_dict={**arg_dict, **{'gamma':gamma, 'C':c}}, grid=True, X=X, Y=Y, n_folds=n_folds, random_state=random_state) for gamma, c in iteration);
	#metrics = [cross_validation.cross_validation(arg_dict={**arg_dict, **{'gamma':gamma, 'C':c}}, grid=True, X=X, Y=Y, n_folds=n_folds, random_state=random_state) for gamma, c in iteration]
	print('Done. Time Elapsed = {}'.format(time.time() - start))
	max_metric = -1.0
	opt_classifier = None
	for m in metrics:
		avg = calc_avg_metrics(m)
		if avg[metric] > max_metric:
			max_metric = avg[metric]
			opt_classifier = m
	return opt_classifier