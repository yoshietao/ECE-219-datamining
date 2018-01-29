from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import sys

class Data:															#all the data from Internet database with diff category sets
	def __init__(self,cat1,cat2,path):
		self.categories1 = cat1
		self.categories2 = cat2
		self.data1 = fetch_20newsgroups(data_home = path,subset='train', categories=cat1, shuffle=True, random_state=42)
		self.data2 = fetch_20newsgroups(data_home = path,subset='train', categories=cat2, shuffle=True, random_state=42)
		self.data3 = fetch_20newsgroups(data_home = path,subset='test', categories=cat1, shuffle=False, random_state=42)
		self.training_data1 = self.data1.data 										# get the data
		self.training_target1 = np.array([int(i>3) for i in self.data1.target])		# get the target--> binary class, so change 8 sub class to 2 class
		#for i in range(len(self.training_target1)):
		#	print (self.training_target1[i],data1.target_names[data1.target[i]])
		self.training_data2 = self.data2.data 										# get the data
		self.training_target2 = self.data2.target 									# get the target
		self.testing_data1 = self.data3.data 										# get the data
		self.testing_target1 = np.array([int(i>3) for i in self.data3.target]) 		# get the target--> binary class, so change 8 sub class to 2 class

def plot_histogram(dclass): 															#part A
	dictt = {}
	for i in dclass.categories1:
		training_data = fetch_20newsgroups(data_home = path,subset='train', categories=[i])
		dictt[i] = len(training_data.data)

	fig,ax = plt.subplots()
	plt.bar(list(dclass.data1.target_names), list(dictt.values()))
	labels = ax.get_xticklabels()
	plt.setp(labels, rotation=20, fontsize=10)
	plt.show()

def preprocess(dclass,data,vectorizer,tfidf_transformer,train=True,ICF=False):		#preprocess(re + TF-IDF)
	data_re = doc_re(data)
	data_tfidf = tfidf(dclass,data_re,vectorizer,tfidf_transformer,train,ICF)		#default ICF=False
	return data_tfidf

def doc_re(data):																	#remove puntuation
	stop_words = text.ENGLISH_STOP_WORDS
	stem = WordNetLemmatizer()
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	filter_punc = [regex.sub('', text) for text in data]
	return [' '.join([stem.lemmatize(word.lower()) for word in doc.split(' ') if word.lower() not in stop_words])for doc in filter_punc]

def tfidf(dclass,doc_re,vectorizer,tfidf_transformer,train,ICF):					#TF-IDF or TF-ICF using "ICF" parameter, default ICF is false
	X = vectorizer.fit_transform(doc_re) if train else vectorizer.transform(doc_re)
	shape = X[0].toarray().shape
	#print (shape)
	if ICF:
		dclass.vocabulary = {i[1]:i[0] for i in vectorizer.vocabulary_.items()}
		#print (self.vocabulary)
		A = np.zeros([20,shape[1]])						#A is not sparse
		for ind,count in enumerate(X):
			index = dclass.training_target2[ind]
			A[index] += count
		X=A
		del A
	return tfidf_transformer.fit_transform(X) if train else tfidf_transformer.transform(X)

def find_10most(dclass,doc):													#find most significant terms
	doc = doc.toarray()
	col = list(range(doc.shape[1]))
	index = [3,4,6,15]
	t = []
	for i in index:
		t.append([x for x in zip(doc[i],col)])
	#print (t,len(t),len(t[0]))

	max_10 = np.array([sorted(i,key=lambda x:x[0],reverse=True) for i in t])[:,:10]
	
	for id_,i in enumerate(index):
		print (dclass.data2.target_names[i],':',[dclass.vocabulary[int(j[1])] for j in max_10[id_]])

def part_e(dclass,D,Dtest):															#SVD using C=1000 and 0.001
	clf_1000 = SVC(C=1000, probability=True, kernel='linear')
	clf_01   = SVC(C=0.001, probability=True, kernel='linear')
	
	target   = [i*2-1 for i in dclass.training_target1]
	clf_1000.fit(D, target) 
	clf_01.fit(D, target)

	pred_1000 = clf_1000.predict_proba(Dtest)
	pred_01 = clf_01.predict_proba(Dtest)

	ROC_CON_ACC_REC_PRE(pred_1000,dclass.testing_target1)
	ROC_CON_ACC_REC_PRE(pred_01,dclass.testing_target1)

def part_f(dclass,D,Dtest):
	C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	max_ = 0
	c_max = 0
	for i in C:
		clf = SVC(C = i, kernel='linear')
		scores = sum(cross_val_score(clf, D, dclass.training_target1, cv=5))/5
		print (scores)
		c_max = i if scores > max_ else c_max
		max_  = scores if scores > max_ else max_
	print ('best C =',c_max,',which has cross validation score:',max_)

	clf = SVC(C=c_max, probability=True, kernel='linear')
	target   = [i*2-1 for i in dclass.training_target1]
	clf.fit(D, target)
	pred = clf.predict_proba(Dtest)
	ROC_CON_ACC_REC_PRE(pred,dclass.testing_target1)

def part_g(dclass,D,Dtest):
	clf = MultinomialNB()
	clf.fit(D, dclass.training_target1)
	pred = clf.predict_proba(Dtest)
	
	ROC_CON_ACC_REC_PRE(pred,dclass.testing_target1)

def part_h(dclass,D,Dtest):
	clf = LogisticRegression(C=1000)
	clf.fit(D,dclass.training_target1)
	pred = clf.predict_proba(Dtest)
	
	ROC_CON_ACC_REC_PRE(pred,dclass.testing_target1)

def part_i(dclass,D,Dtest):
	error1 = []
	error2 = []
	avg_w1 = []
	avg_w2 = []
	c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	for c in c_list:
		clf1 = LogisticRegression(penalty='l1',C=c)
		clf2 = LogisticRegression(penalty='l2',C=c)
		clf1.fit(D,dclass.training_target1)
		clf2.fit(D,dclass.training_target1)
		pred1 = clf1.predict_proba(Dtest)
		pred2 = clf2.predict_proba(Dtest)
		error1.append( sum([int(int(i>0.5)!=j) for i,j in zip(pred1[:,1],dclass.testing_target1)])/len(dclass.testing_target1) )
		error2.append( sum([int(int(i>0.5)!=j) for i,j in zip(pred2[:,1],dclass.testing_target1)])/len(dclass.testing_target1) )
		avg_w1.append(sum(np.abs(clf1.coef_[0]))/len(clf1.coef_[0]))
		avg_w2.append(sum(np.abs(clf2.coef_[0]))/len(clf2.coef_[0]))
	
	print ('error1',error1,'\nerror2',error2)
	max_1 = c_list[np.argmin(np.array(error1))]
	max_2 = c_list[np.argmin(np.array(error2))]

	clf1 = LogisticRegression(penalty='l1',C=max_1)
	clf2 = LogisticRegression(penalty='l2',C=max_2)

	clf1.fit(D,dclass.training_target1)
	clf2.fit(D,dclass.training_target1)
	
	pred1 = clf1.predict_proba(Dtest)
	pred2 = clf2.predict_proba(Dtest)

	ROC_CON_ACC_REC_PRE(pred1,dclass.testing_target1)
	ROC_CON_ACC_REC_PRE(pred2,dclass.testing_target1)
	print ('avg weight:',avg_w1,'\n',avg_w2)

def plot_ROC(pred_proba,target):

	fpr, tpr, thresholds = roc_curve(target, pred_proba[:,1])
	plt.plot(fpr,tpr)
	plt.title('roc_curve')
	plt.ylabel('TPR')
	plt.xlabel('FPR')
	plt.show()

def acc_rec_pre(y_true,y_pred):
	accuracy  = 0
	correct_1 	  = 0
	for i,j in zip(y_true,y_pred):
		accuracy = accuracy+1 if i==j else accuracy
		correct_1  = correct_1+1 if i==j==1 else correct_1
	accuracy /=len(y_true)
	precision = correct_1/sum(y_pred) if sum(y_pred) != 0 else 0
	recall = correct_1/sum(y_true)
	print ('accuracy/precision/recall=',accuracy,precision,recall)

def ROC_CON_ACC_REC_PRE(pred_proba,y_true):
	plot_ROC(pred_proba,y_true)
	y_pred = [int(i[1]>0.5) for i in pred_proba]
	print (confusion_matrix(y_true, y_pred))
	acc_rec_pre(y_true,y_pred)

def main(argv):
	choose_mindf = argv[1]
	try:
		path = argv[2]
	except:
		path = None
	categories1 = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
	categories2 = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
	cat_all = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian', 'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
	dclass = Data(categories1,cat_all,path)
	stop_words = text.ENGLISH_STOP_WORDS

	print ('-----Part A-----')
	#plot_histogram(dclass)
	print ('-----Part B-----')

	vectorizer2 = CountVectorizer(min_df=2,stop_words=stop_words,max_df=0.8)
	tfidf_transformer2 = TfidfTransformer()

	vectorizer5 = CountVectorizer(min_df=5,stop_words=stop_words,max_df=0.8)
	tfidf_transformer5 = TfidfTransformer()
	tfidf2 = preprocess(dclass,dclass.training_data1,vectorizer2,tfidf_transformer2,train=True)
	tfidf5 = preprocess(dclass,dclass.training_data1,vectorizer5,tfidf_transformer5,train=True)						#default min_df=5

	print ('# of terms with min_df = 2:',tfidf2[0,:].toarray().shape[1],'\n# of terms with min_df = 5:',tfidf5[0,:].toarray().shape[1])

	d_tfidf = {'2':tfidf2,'5':tfidf5}
	d_vectorizer = {'2':vectorizer2,'5':vectorizer5}
	d_transformer = {'2':tfidf_transformer2,'5':tfidf_transformer5}
	
	print ('-----Part C-----')

	#vectorizerc = CountVectorizer(min_df=int(choose_mindf),stop_words=stop_words,max_df=0.8)
	#tfidf_transformerc = TfidfTransformer()


	vectorizerc_2 = CountVectorizer(min_df=2,stop_words=stop_words,max_df=0.8)
	tfidf_transformerc_2 = TfidfTransformer()

	tfidf_c_2 = preprocess(dclass,dclass.training_data2,vectorizerc_2,tfidf_transformerc_2,train=True,ICF=True)			#default min_df=5, use TF-ICF
	find_10most(dclass,tfidf_c_2)

	vectorizerc_5 = CountVectorizer(min_df=5,stop_words=stop_words,max_df=0.8)
	tfidf_transformerc_5 = TfidfTransformer()

	tfidf_c_5 = preprocess(dclass,dclass.training_data2,vectorizerc_5,tfidf_transformerc_5,train=True,ICF=True)			#default min_df=5, use TF-ICF
	find_10most(dclass,tfidf_c_5)


	
	print ('-----Part D-----')																						#SVD and NMF base on TF-IDF5 result
	svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
	D_LSI = svd.fit_transform(d_tfidf[choose_mindf])
	model = NMF(n_components=50, init='random', random_state=0)
	D_NMF = model.fit_transform(d_tfidf[choose_mindf])
	print ('LSI.shape:',D_LSI.shape,'\nNMF.shape:',D_NMF.shape)

	print ('-----Part E-----')																						#SVM
	tfidftest = preprocess(dclass,dclass.testing_data1,d_vectorizer[choose_mindf],d_transformer[choose_mindf],train=False)					#testing data
	D_LSI_test = svd.transform(tfidftest)
	D_NMF_test = model.transform(tfidftest)
	'''
	print ('for D_LSI:')
	part_e(dclass,D_LSI,D_LSI_test)
	print ('for D_NMF:')
	part_e(dclass,D_NMF,D_NMF_test)

	print ('-----Part F-----')
	print ('for D_LSI:')
	part_f(dclass,D_LSI,D_LSI_test)
	print ('for D_NMF:')
	part_f(dclass,D_NMF,D_NMF_test)
	
	print ('-----Part G-----')
	part_g(dclass,D_NMF,D_NMF_test)
	
	print ('-----Part H-----')
	part_h(dclass,D_LSI,D_LSI_test)
	part_h(dclass,D_NMF,D_NMF_test)
	'''
	print ('-----Part I-----')
	part_i(dclass,D_LSI,D_LSI_test)
	part_i(dclass,D_NMF,D_NMF_test)

	print ('-----Part J-----')
	



	#####################
	#Left: problem J
	#####################

if __name__ == '__main__':
	main(sys.argv)
