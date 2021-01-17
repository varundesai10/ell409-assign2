import numpy as np
import random as rnd

class SVM():
	def __init__(self, kernel='linear',C=1.0, gamma='auto', degree=2, max_iter=10, eps=1e-3):

		
		self.kernel_type = kernel
		self.C = C
		self.gamma = gamma
		self.degree = degree
		self.max_iter = max_iter
		self.eps = eps

	def fit(self, X, y):
		n, d = X.shape[0], X.shape[1]
		y[y==0] = -1
		assert(X.shape[0] == y.shape[0]), "X and y must have the same number of samples!"

		if self.kernel_type=='linear':
			self.kernel = lambda x, y: np.dot(x, y)
		
		elif self.kernel_type == 'poly':
			if self.gamma == 'auto':
				self.gamma = 1/(d*X.var())
			self.kernel = lambda x, y: np.pow(1 + self.gamma*np.dot(x, y), self.degree)
		
		elif self.kernel_type == 'rbf':
			if self.gamma == 'auto':
				self.gamma = 1(d*X.var())
			self.kernel = lambda x, y: np.exp(-self.gamma*np.norm(x-y)**2)

		self.alpha = np.zeros(n)
		self.b = 0
		self.K = np.zeros((n, n))

		#initializing kernel matrix
		for i in range(n):
			for j in range(n):
				self.K[i, j] = self.kernel(X[i], X[j])

		passes = 0
		#print(self.K)
		while(passes<self.max_iter):
			num_changed_alphas=0
			#print(self.alpha)
			for i in range(n):
				#a = input()
				#iterate over i
				E_i = self.f(i, y) - y[i]
				#print(f'E_i = {E_i}, f(i, y) = {self.f(i, y)}, y = {y[i]}')
				if( (y[i]*E_i < -self.eps and self.alpha[i] < self.C) or (y[i]*E_i > self.eps and self.alpha[i] > 0) ):
					#randomly choose j != i
					#print('Entered Loop')
					j = self.rndint(0, n-1, i)
					E_j = self.f(j, y) - y[j]
					alpha_i_o, alpha_j_o = self.alpha[i], self.alpha[j]

					L, H = self.compute_L_H(alpha_j_o, alpha_i_o, y[j], y[i]) #the bound for alpha_j
					#print(f'L = {L}, H = {H}')
					neta = 2*self.K[i, j] - self.K[i,i] - self.K[j,j]
					#print(f'neta = {neta}')
					if(neta >= 0):
						continue

					self.alpha[j] = self.alpha[j] - y[j]*(E_i - E_j)/neta

					if self.alpha[j] < L:
						self.alpha[j] = L
					elif self.alpha[j] > H:
						self.alpha[j] = H

					if(self.alpha[j] - alpha_j_o < 1e-5):
						continue

					self.alpha[i] = self.alpha[i] + y[i]*y[j]*(alpha_j_o - self.alpha[j])
					b_i = self.b - E_i - y[i]*(self.alpha[i] - alpha_i_o)*self.K[i,i] - y[j]*(self.alpha[j] - alpha_j_o)*self.K[i,j]
					b_j = self.b - E_i - y[i]*(self.alpha[i] - alpha_i_o)*self.K[i,j] - y[j]*(self.alpha[j] - alpha_j_o)*self.K[j,j]

					if(np.abs(self.alpha[i]) > self.eps and np.abs(self.alpha[i]-self.C) > self.eps):
						self.b = b_i
					elif(np.abs(self.alpha[j]) > self.eps and np.abs(self.alpha[i] - self.C) > self.eps):
						self.b = b_j
					else:
						self.b = 0.5*(b_i + b_j)
					num_changed_alphas += 1
					#print(f'Optimised over {i}, {j}')
			
			if(num_changed_alphas == 0):
				passes += 1
			else:
				passes = 0



	
	def compute_L_H(self, alpha_j_o, alpha_i_o, y_j, y_i):
		
		if(y_i != y_j):
			return (max(0, alpha_j_o - alpha_i_o), min(self.C, self.C - alpha_i_o + alpha_j_o))
		else:
			return (max(0, alpha_i_o + alpha_j_o - self.C), min(self.C, alpha_i_o + alpha_j_o))

	def f(self, i, y):
		return np.dot(self.alpha*y, self.K[:, i]) + self.b

	def rndint(self, a,b,z):
		i = z
		cnt=0
		while i == z and cnt<1000:
			i = rnd.randint(a,b)
			cnt=cnt+1
		return i



