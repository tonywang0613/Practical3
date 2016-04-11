import numpy as np
import csv
import pickle
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt

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
	
	def	__init__(self,alpha=1e-4,la=0.1):
		
		self.alpha=alpha
		self.la=la
		
		#load train data set, three column, id_index of user, id_index of artist,plays
		with open('train.pickle', 'rb') as handle:
			self.train = pickle.load(handle)
		
		self.train = np.array(self.train, dtype = 'float_')
		
		#convert the plays to z score
		plays=self.train[:,2]
		self.mean=np.median(plays)
		self.std=np.std(plays)
		
		min=np.min(plays)
		max=np.max(plays)
		
		print "min {},max{},mean {},std {}".format(min,max,self.mean,self.std)
		error=np.sum(np.abs(plays-self.mean))/len(plays)
		print "mean loss{}".format(error)
		
		self.train[:,2]=(self.train[:,2]-self.mean)/self.std
		#calculate global med:
		'''
		d=defaultdict(list)
		for i in self.train:
			 d[i[0]].append(i[2])
		for (user,score) in d.items():
			d[user]=np.median(np.asarray(score))
			 
		scores=np.asarray([d[i[0]] for i in self.train])
		new_column=scores.reshape(len(scores),1)
		
		self.train=np.hstack((self.train,new_column))
		'''
		#load test data set, two column, id_index of user, id_index of artist
		with open('test.pickle','rb') as handle:
			test=pickle.load(handle) 
	
		self.num_users=233286
		self.num_artist=2000
		self.features=3
		
		
		'''
		This method is too slow
		#convert train data into matrix, row is artist, column is the user
		#m[i][j] is the plays of j user with i artist
		
		index=np.arange(self.num_artist)
		column=np.arange(self.num_users)
		
		self.df=pd.DataFrame(index=index,columns=column)
		#print self.df
		#populate the training value 
		for row in self.train:
			i=int(row[1])
			j=int(row[0])
			#print i,j
			self.df[i][j]=row[2]
			#print self.df[i][j]
		
		print "df created"
		'''
		
		#create two matrix 
		self._user_features=0.4*np.random.rand(self.num_users,self.features)
		self._artist_features=0.4*np.random.rand(self.num_artist,self.features)
	
	def draw(self):
		with open('erro_li_p3','rb') as handle:
			errors=pickle.load(handle)
		x=[]
		y=[]
		for idx, val in enumerate(errors):
			x.append(idx)
			y.append(val)
		plt.plot(x,y)
		plt.show()
		
		
	def estimate(self,iterations=500,converge=1e-7):
		last_loss=None
		data=self.train
		
		_user_features=self._user_features
		_artist_features=self._artist_features
		
		loss_li=[]
		user_li=[]
		artist_li=[]
		
		for iteration in range(iterations):
			
			
			#calculate the gradient
			#update the _user_features first
			
			#compute gradien
			
			u_features = self._user_features[data[:, 0].astype(int), :]
			i_features = self._artist_features[data[:, 1].astype(int), :]
			
			# print "u_feature", u_features
			# print "i_feature", i_features.shape
			
			preds = np.sum(u_features * i_features, 1)
			
			
			'''
			new_column=preds.reshape((preds.shape[0],1))
			print np.asarray(preds).transpose()
			print "data shape{},preds shape {}".format(data.shape,new_column.shape)
			data=np.hstack((data,new_column))
			#print "ee"
			user_id=0
			
			#update the _user_features 
			for u in range(_user_features.shape[0]):
				
				user_id=u
				#select from data
				new_data=data[np.where(data[:,0]==user_id)]
				#y_hat of this user
				y_hat_u=new_data[:,2]
				#artist_feature
				artist_feature_u=_artist_features[new_data[:,1].astype(int),:]
				part1=np.sum(artist_feature_u*_user_features[u],1)-y_hat_u
				
				sum_error=np.sum(np.sum(np.sum(artist_feature_u*_user_features[u],1)-y_hat_u)*_user_features[u])
				
				#updata the _user_features
				_user_features[u]-=self.alpha*(sum_error+self.la*_user_features[u])
			
			
			#update the _artist_features
			for i in range(_artist_features.shape[0]):
				
				artist_id=i
				new_data=data[np.where(data[:,1]==artist_id)]
				y_hat_i=new_data[:,2]
				user_feature_i=_user_features[new_data[:,0].astype(int),:]
				
				sum_error=np.sum(np.sum(np.sum(user_feature_i*_artist_features[i],1)-y_hat_i)*_artist_features[i])	
				_artist_features[i]-=self.alpha*(sum_error+self.la*_artist_features[i])
			
			self._user_features=_user_features
			self._artist_features=_artist_features
			'''
			errs = preds - data[:,2]
			#print errs
			err_mat = np.tile(errs, (self.features, 1)).T
			#print "err_mat",err_mat.shape
			
			u_grads = i_features * err_mat + self.la * u_features
			i_grads = u_features * err_mat + self.la * i_features
			
			u_feature_grads = np.zeros((self.num_users, self.features))
			i_feature_grads = np.zeros((self.num_artist, self.features))
			
			
			for i in xrange(data.shape[0]):
				user = data[i, 0]
				artist = data[i, 1]
				u_feature_grads[user, :] += u_grads[i,:]
				i_feature_grads[artist, :] += i_grads[i,:]
			
			# update latent variables
			print self.alpha * u_feature_grads
			self._user_features = self._user_features - \
				self.alpha * u_feature_grads
			self._artist_features = self._artist_features - \
				self.alpha * i_feature_grads
			
			

			
			
			
			train_preds=self.predict(data)
			train_loss=self.calc_loss(train_preds)
			print train_loss,last_loss
			
			if last_loss:
				print train_loss-last_loss
				if abs(train_loss-last_loss)<converge:
					break
			last_loss=train_loss
			
			loss_li.append(train_loss)
			user_li.append(self._user_features)
			artist_li.append(self._artist_features)
			with open('erro_li_p3', 'wb') as handle:
				pickle.dump(loss_li, handle)
			with open('user_li_p3', 'wb') as handle:
			  pickle.dump(user_li, handle)
			with open('artist_li_p3', 'wb') as handle:
			  pickle.dump(artist_li, handle)
	#calucate one of the gradient component		

		
	def get_y(self,i,j,data):
		data=self.train
		
	def calc_mean(self,train):
		return np.mean(train[:,2])
		
	def calc_std(self,train):
		return np.std(train[:,2])
		
	#data contains two column, user id_index, and artist id_index
	def predict(self,data):
		u_features=self._user_features[data[:,0].astype(int),:]
		i_features=self._artist_features[data[:,1].astype(int),:]
		
		preds=np.sum(u_features*i_features,1)
		
		preds[preds>1]=1
		preds[preds<-1]=-1
		
		return preds
		
	#based on the function by Andrew Ng video
	def calc_loss(self, preds):
		"""Root Mean Square Error"""
		
		truth=np.float16(self.train[:,2])
		
		

		num_sample = len(preds)

		# sum square error 
		sse = np.sum(np.abs(self.std*(truth - preds)))
		print "sse{} num_samples{}".format(sse,num_sample)
		return np.divide(sse, num_sample)
	
if __name__ == "__main__":
	
	pmf=pmfimple()
	pmf.draw()
	#pmf.estimate() 
