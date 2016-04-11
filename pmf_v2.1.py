import numpy as np
import csv
import pickle
import pandas as pd
from numpy import linalg as LA

class pmfimple:
	
	def load_data(self):
	
		train_file = 'train.csv'
		test_file  = 'test.csv'
		soln_file  = 'user_median.csv'

		user_profile="profiles.csv"
		artist="artist.csv"

		train=[]
		test=[]
		user_dic_ind={}
		artist_dic_ind={}
		user_index=0
		artist_index=0
		with open(train_file,'r') as train_fh:
			train_csv =csv.reader(train_fh, delimiter=',', quotechar='"')
			next(train_csv, None)
			for row in train_csv:
				user,artist,plays=row[0],row[1],row[2]
		
				
				if user not in user_dic_ind:
					user_dic_ind[user]=user_index
					user_index+=1
		
				
				if artist not in artist_dic_ind:
					artist_dic_ind[artist]=artist_index
					artist_index+=1
			
				#create train data set , with three column , user id_index (0-?), artist id_index (0-?) and plays
		
		
		
				train.append([user_dic_ind[user],artist_dic_ind[artist],plays])
	
			train=np.asarray(train)
			#print train.shape
			#print user_index,artist_index
	
		with open(test_file,"r") as test_fh:
			test_csv=csv.reader(test_fh,delimiter=',',quotechar='"')
			next(test_csv,None)
	
			for row in test_csv:
				user,artist=row[1],row[2]
				if user not in user_dic_ind:
					user_dic_ind[user]=user_index
					user_index+=1
					
				if artist not in artist_dic_ind:
					artist_dic_ind[artist]=artist_index
					artist_index+=1
					
				# create test data set, with two column, user_index, artist_index
				test.append([user_dic_ind[user],artist_dic_ind[artist]])
				
			test=np.asarray(test)
			
		
		print "user_num {}, artist_num {}".format(user_index,artist_index)
		#save test and train data set 
		with open('train.pickle', 'wb') as handle:
		  pickle.dump(train, handle)
		with open('test.pickle', 'wb') as handle:
		  pickle.dump(test, handle)
	
	def	__init__(self,alpha=4e-6,la=0.001):
		
		self.alpha=alpha
		self.la=la
		
		#load train data set, three column, id_index of user, id_index of artist,plays
		with open('train.pickle', 'rb') as handle:
			self.train = pickle.load(handle)
		
		self.train = np.array(self.train, dtype = 'float_')
		
		#convert the plays to z score
		plays=self.train[:,2]
		self.mean=np.mean(plays)
		self.std=np.std(plays)
		
		min=np.min(plays)
		max=np.max(plays)
		
		print "min {},max{},mean {},std {}".format(min,max,self.mean,self.std)
		
		self.train[:,2]=(self.train[:,2]-self.mean)/self.std
		
		num_users=233286
		num_artist=2000
		
		
		small_test=False
		if small_test==True:
			users=[]
			artist=[]
			self.train=self.train[:1e+4,]
			
			for i in self.train:
				if i[0] not in users:
					users.append(i[0])
				if i[1] not in artist:
					artist.append(i[1])
			num_users=len(users)
			num_artist=len(artist)
			
		#load test data set, two column, id_index of user, id_index of artist
		with open('test.pickle','rb') as handle:
			test=pickle.load(handle) 
		
		self.num_users=num_users
		self.num_artist=num_artist
		self.features=5
		
		
	
		
		#convert train data into matrix, row is artist, column is the user
		#m[i][j] is the plays of j user with i artist
		
		X=np.zeros((self.num_users,self.num_artist))
		
		for i in self.train:
			X[i[0]][i[1]]=i[2]
		
		
		self.X=X
		print X.shape
		#populate the training value 

		print "df created"
		
	def estimate(self,K):
		X=self.X
		#num of users
		N=X.shape[0]
		#num of artist
		M=X.shape[1]
		P=np.random.rand(N,K)
		Q=np.random.rand(M,K)
		steps = 5000
		alpha = 0.02
		beta = float(0.02)
		estimated_P, estimated_Q =self.matrix_factorization(X,P,Q,K,steps,alpha,beta)
		
	def matrix_factorization(self,X,P,Q,K,steps,alpha,beta):
		Q = Q.T
		erro_li=[]
		P_li=[]
		Q_li=[]
		for step in xrange(steps):
			print step
			#for each user
			for i in xrange(X.shape[0]):
		            #for each item
				for j in xrange(X.shape[1]):
					if X[i][j] > 0 :

		                    #calculate the error of the element
						eij = X[i][j] - np.dot(P[i,:],Q[:,j])
		                    #second norm of P and Q for regularilization
						sum_of_norms = 0
		                    #for k in xrange(K):
		                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
		                    #added regularized term to the error
						sum_of_norms += LA.norm(P) + LA.norm(Q)
		                    #print sum_of_norms
						eij += ((beta/2) * sum_of_norms)
		                    #print eij
		                    #compute the gradient from the error
						for k in xrange(K):
							P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
							Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))
				#print "j {}".format(j)
			print "i {}".format(i)

		        #compute total error
			error = 0
		        #for each user
			for i in xrange(X.shape[0]):
		            #for each item
				for j in xrange(X.shape[1]):
					if X[i][j] > 0:
						error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
		 				if error < 0.001:
							break
			
			erro_li.append(error)
			P_li.append(P)
			Q_li.append(Q)
			with open('erro_li_v21', 'wb') as handle:
				pickle.dump(erro_li, handle)
			with open('P_li_v21', 'wb') as handle:
			  pickle.dump(P_li, handle)
			with open('Q_li_v21', 'wb') as handle:
			  pickle.dump(Q_li, handle)
			
			
			print step,error
		return P, Q.T
		
if __name__ == "__main__":
	
	pmf=pmfimple()
	pmf.estimate(3) 

