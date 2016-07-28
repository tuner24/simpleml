# coding:utf-8

"""
hmm1.py is two slow: delta_lambda = 15 ,x=8 more zhan 30k times
1. with scale
2. be modified

References:
1. 统计学习方法
2. 数学之美
3. http://www.tuicool.com/articles/3iENzaV

Pivot:
1. alpha,beta,gamma,xi are all for Baum_Welch
2. delta,psi are for viterbi
3. Baum_Welch for traning, viterbi for predict

Ideas for application:
1. predict 

2. speech recognition

3. NLP 

error:
1. the computing X must be array,  need to np.array
2. and the cells must be float, need np.float, if initialization need decimal.
3. initialization is big proplem, every cell shouldn't be 0
"""

import numpy as np
from scipy.spatial.distance import cdist   # computing Chebyshev distance between matrixes

class HMM:
	def __init__(self, Ann, Bnm, Pi, O):
		self.A = np.array(Ann, np.float)
		self.B = np.array(Bnm, np.float)
		self.Pi = np.array(Pi, np.float)
		self.O = np.array(O, np.float)
		self.N = self.A.shape[0]
		self.M = self.B.shape[1]


	def forward(self):
		T = len(self.O)
		alpha = np.zeros((T, self.N), np.float)

		for i in range(self.N):		
			alpha[0,i] = self.Pi[i] * self.B[i, self.O[0]]

		for t in range(T-1):
			for i in range(self.N):
				summation = 0   # for every i 'summation' should reset to '0'
				for j in range(self.N):
					summation += alpha[t,j] * self.A[j,i]
				alpha[t+1, i] = summation * self.B[i, self.O[t+1]]

		summation = 0.0
		for i in range(self.N):
			summation += alpha[T-1, i]
		Polambda = summation
		return Polambda,alpha

	def forward_with_scale(self):
		T = len(self.O)
		alpha_raw = np.zeros((T, self.N), np.float)
		alpha = np.zeros((T, self.N), np.float)
		c = [i for i in range(T)]  # scalint factor; 0 or sequence doesn't matter

		for i in range(self.N):		
			alpha_raw[0,i] = self.Pi[i] * self.B[i, self.O[0]]
			
		c[0] = 1.0 / sum(alpha_raw[0,i] for i in range(self.N))
		for i in range(self.N):
			alpha[0, i] = c[0] * alpha_raw[0,i]

		for t in range(T-1):
			for i in range(self.N):
				summation = 0.0
				for j in range(self.N):
					summation += alpha[t,j] * self.A[j, i]
				alpha_raw[t+1, i] = summation * self.B[i, self.O[t+1]]

			c[t+1] = 1.0 / sum(alpha_raw[t+1,i1] for i1 in range(self.N))
			
			for i in range(self.N):
				alpha[t+1, i] = c[t+1] * alpha_raw[t+1, i]
		return alpha, c


	def backward(self):
		T = len(self.O)
		beta = np.zeros((T, self.N), np.float)
		for i in range(self.N):
			beta[T-1, i] = 1.0

		for t in range(T-2,-1,-1):
			for i in range(self.N):
				summation = 0.0     # 这个必须在这一行，每一次for i 都要重置为0
				for j in range(self.N):
					summation += self.A[i,j] * self.B[j, self.O[t+1]] * beta[t+1,j]
				beta[t,i] = summation

		Polambda = 0.0
		for i in range(self.N):
			Polambda += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]
		return Polambda, beta

	def backward_with_scale(self,c):
		T = len(self.O)
		beta_raw = np.zeros((T, self.N), np.float)
		beta = np.zeros((T, self.N), np.float)
		for i in range(self.N):
			beta_raw[T-1, i] = 1.0
			beta[T-1, i] = c[T-1] * beta_raw[T-1, i]

		for t in range(T-2,-1,-1):
			for i in range(self.N):
				summation = 0.0
				for j in range(self.N):
					summation += self.A[i,j] * self.B[j, self.O[t+1]] * beta[t+1,j]
				beta[t,i] = c[t] * summation   # summation = beta_raw[t,i]
		return beta


	def viterbi(self):
		# given O,lambda .finding I

		T = len(self.O)
		I = np.zeros(T, np.float)

		delta = np.zeros((T, self.N), np.float)  
		psi = np.zeros((T, self.N), np.float)

		for i in range(self.N):
			delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
			psi[0, i] = 0

		for t in range(1, T):
			for i in range(self.N):
				delta[t, i] = self.B[i,self.O[t]] * np.array( [delta[t-1,j] * self.A[j,i]
					for j in range(self.N)] ).max() 
				psi[t,i] = np.array( [delta[t-1,j] * self.A[j,i] 
					for j in range(self.N)] ).argmax()

		P_T = delta[T-1, :].max()
		I[T-1] = delta[T-1, :].argmax()

		for t in range(T-2, -1, -1):
			I[t] = psi[t+1, I[t+1]]

		return I

	def compute_gamma(self,alpha,beta):
		T = len(self.O)
		gamma = np.zeros((T, self.N), np.float)       # the probability of Ot=q
		for t in range(T):
			for i in range(self.N):
				gamma[t, i] = alpha[t,i] * beta[t,i] / sum(
					alpha[t,j] * beta[t,j] for j in range(self.N) )
		return gamma

	def compute_xi(self,alpha,beta):
		T = len(self.O)
		xi = np.zeros((T-1, self.N, self.N), np.float)  # note that: not T
		for t in range(T-1):   # note: not T
			for i in range(self.N):
				for j in range(self.N):
					numerator = alpha[t,i] * self.A[i,j] * self.B[j,self.O[t+1]] * beta[t+1,j]
					# the multiply term below should not be replaced by 'nummerator'，
					# since the 'i,j' in 'numerator' are fixed.
					# In addition, should not use 'i,j' below, to avoid error and confusion.
					denominator = sum( sum(	 
						alpha[t,i1] * self.A[i1,j1] * self.B[j1,self.O[t+1]] * beta[t+1,j1] 
						for j1 in range(self.N) )   # the second sum
							for i1 in range(self.N) )	# the first sum
					xi[t,i,j] = numerator / denominator
		return xi

	def Baum_Welch(self):
		# given O list finding lambda model(can derive T form O list)
		# also given N, M, 
		T = len(self.O)
		V = [k for k in range(self.M)]

		# initialization - lambda 
		self.A = np.array(([[0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5]]), np.float)
		self.B = np.array(([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]]), np.float)

		# mean value is not a good choice
		self.Pi = np.array(([1.0 / self.N] * self.N), np.float)  # must be 1.0 , if 1/3 will be 0
		# self.A = np.array([[1.0 / self.N] * self.N] * self.N) # must array back, then can use[i,j]
		# self.B = np.array([[1.0 / self.M] * self.M] * self.N)

		x = 1
		delta_lambda = x + 1
		times = 0
		# iteration - lambda
		while delta_lambda > x:  # x
			Polambda1, alpha = self.forward()   		# get alpha
			Polambda2, beta = self.backward()			# get beta
			gamma = self.compute_gamma(alpha,beta)     # use alpha, beta
			xi = self.compute_xi(alpha,beta)

			lambda_n = [self.A,self.B,self.Pi]

			
			for i in range(self.N):
				for j in range(self.N):
					numerator = sum(xi[t,i,j] for t in range(T-1))
					denominator = sum(gamma[t,i] for t in range(T-1))
					self.A[i, j] = numerator / denominator

			for j in range(self.N):
				for k in range(self.M):
					numerator = sum(gamma[t,j] for t in range(T) if self.O[t] == V[k] )  # TBD
					denominator = sum(gamma[t,j] for t in range(T))
					self.B[i, k] = numerator / denominator

			for i in range(self.N):
				self.Pi[i] = gamma[0,i]

			# if sum directly, there will be positive and negative offset
			# computes the Chebyshev distance between two matrixes
			cdist_A = cdist(lambda_n[0], self.A, 'chebyshev')  # cdist_A is still a matrix
			cdist_B = cdist(lambda_n[1], self.B, 'chebyshev')
			cdist_Pi = cdist([lambda_n[2]], [self.Pi], 'chebyshev') # turn Pi(vector) to matrix.
			delta_lambda = sum([ sum(sum(cdist_A)), sum(sum(cdist_B)), sum(sum(cdist_Pi)) ])
			times += 1
			print times

		return self.A, self.B, self.Pi


	def Baum_Welch_with_scale(self):
		T = len(self.O)
		V = [k for k in range(self.M)]

		# initialization - lambda   ,  should be float(need .0)
		self.A = np.array([[0.2,0.2,0.3,0.3],[0.2,0.1,0.6,0.1],[0.3,0.4,0.1,0.2],[0.3,0.2,0.2,0.3]])
		self.B = np.array([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]])

		x = 3
		delta_lambda = x + 1
		times = 0
		# iteration - lambda
		while delta_lambda > x:  # x
			alpha,c = self.forward_with_scale()
			beta = self.backward_with_scale(c)			
			gamma = self.compute_gamma(alpha,beta)   
			xi = self.compute_xi(alpha,beta)

			lambda_n = [self.A,self.B,self.Pi]
			
			for i in range(self.N):
				for j in range(self.N):
					numerator = sum(xi[t,i,j] for t in range(T-1))
					denominator = sum(gamma[t,i] for t in range(T-1))
					self.A[i, j] = numerator / denominator

			for j in range(self.N):
				for k in range(self.M):
					numerator = sum(gamma[t,j] for t in range(T) if self.O[t] == V[k] )  # TBD
					denominator = sum(gamma[t,j] for t in range(T))
					self.B[i, k] = numerator / denominator

			for i in range(self.N):
				self.Pi[i] = gamma[0,i]

			# if sum directly, there will be positive and negative offset
			# computes the Chebyshev distance between two matrixes
			cdist_A = cdist(lambda_n[0], self.A, 'chebyshev') 
			cdist_B = cdist(lambda_n[1], self.B, 'chebyshev')
			cdist_Pi = cdist([lambda_n[2]], [self.Pi], 'chebyshev') 
			# delta_lambda = sum([ sum(sum(cdist_A)), sum(sum(cdist_B)), sum(sum(cdist_Pi)) ])
			# try igore B's error, not work casue the lambda is still in local optimal
			delta_lambda = sum([ sum(sum(cdist_A)), sum(sum(cdist_Pi)) ])
			times += 1
			print times

		return self.A, self.B, self.Pi

def try_viterbi():
	A11 = [[0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5]]
	B11 = [[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]]
	Pi11 = [0.25] * 4
	O11 = [0,1,0,1,0,0,1,0,1]   # 0 denote red, 1 denote white
	hmm11 = HMM(A11, B11, Pi11, O11)
	I = hmm11.viterbi()
	print I

def try_Baum_Welch():
	A21 = np.zeros((4,4), np.float)
	B21 = np.zeros((4,2), np.float)    
	Pi21 = [0.25,0.25,0.25,0.25]   					    	
	O21 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
	V21 = [k for k in range(2)]
	Q21 = [q for q in range(4)]
	# T21 = len(O21)
	hmm21 = HMM(A21, B21, Pi21, O21)

	A, B, Pi = hmm21.Baum_Welch()
	print A ,B ,Pi

def try_Baum_Welch_with_scale():
	A22 = np.zeros((4,4), np.float)
	B22 = np.zeros((4,2), np.float)    
	Pi22 = [0.25,0.25,0.25,0.25]   					    	
	O22 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 4
	# V22 = [k for k in range(2)]
	# Q22 = [q for q in range(4)]
	# T21 = len(O21)
	hmm22 = HMM(A22, B22, Pi22, O22)

	A, B, Pi = hmm22.Baum_Welch_with_scale()
	print A ,B ,Pi


if __name__ == '__main__':
	print 'HMM in python'
	"""
	# Lambda: A = N * N -ij, B = N * M -jk, Pi = N -i
	# Q-N-qj ; V-M-vk ; 
	# State sequence: I-T-it . Observation sequence: O-T-ot .
	"""
	# for viterbi : lambda O - I
	# A1 = 
	# B1 = 
	# Pi1 = 
	# O1 = 
	# hmm1 = HMM(A1, B1, Pi1, O1)
	# I = hmm1.viterbi
	# ------ try 
	# try_Baum_Welch()
	try_Baum_Welch_with_scale()

	# for Baum_Welch : O-lambda
	# A2 = np.zeros((N,N), np.float)
	# B2 = np.zeros((N,M), np.float)
	# Pi2 = np.zeros(N, np.float)
	# O2 = 
	# ...
	# V2 = [k for k in range(M)]
	# Q2 = [q for q in range(N)]
	# T = len(O2)
	# I = np.zeros(T)


	# alpha = np.zeros((T, hmm1.N), np.float)
	# beta = np.zeros((T, hmm1.N), np.float)

	# gamma = np.zeros((T, hmm1.N), np.float)       # the probability of Ot=q
	# xi = np.zeros((T, hmm1.N, hmm1.N), np.float)  # refer to: pg179- 10.25

	# hmm2 = HMM(A2, B2, Pi2, O2)
	# A, B, Pi = hmm2.Baum_Welch

	# ------ try 



