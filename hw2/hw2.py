from sklearn.datasets import fetch_20newsgroups
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


class Data:
	def __init__(self,cat):
		data = fetch_20newsgroups(subset='all', categories=cat)
		self.data = data.data
		self.target = data.target
		
def PART(cat):
	d = Data(cat)
	cats = len(cat)
	if len(cat) == 8:
		d.target = np.array([int(i>3) for i in d.target])
		cats = 2
	
	print ('_____1_____')
	data_tfidf = proprocess(d.data)
	print (data_tfidf.shape)
	print ('_____2a_____')
	pred_ = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_tfidf)
	print ('_____2b_____')
	print (print_5_measure(d.target,pred_))
	print ('_____3a_i_____')
	'''
	data_svd = TruncatedSVD(n_components=1000, algorithm='arpack', random_state=42)fit_transform(data_tfidf)
	#data_svd = svd.fit_transform(data_tfidf)
	XXT = data_tfidf*np.transpose(data_tfidf)
	original_variance = 0
	for i in range(XXT.shape[0]):
		original_variance += XXT[i,i]
	sum_var = 0
	variance_retain_ratio_list = []
	for s in svd.singular_values_:
		sum_var += s**2
		variance_retain_ratio_list.append(sum_var/original_variance)
	plt.plot(np.arange(1,1001),variance_retain_ratio_list)
	plt.title('Variance Ratio')
	plt.xlabel('r')		#rank
	plt.show()
	print ('_____3a_ii_____')
	print ('_____SVD_____')
	r_list = [1,2,3,5,10,20,50,100,300]
	measures = []
	for r in r_list:
		svd_r = data_svd[:,:r]
		pred_ = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(svd_r)
		x,y = (print_5_measure(d.target,pred_))
		measures.append(x)
		print (x,y)
	measures = np.array(measures)
	for i in range(5):
		plt.plot(r_list,measures[:,i])
		plt.show()
	print ('______NMF_____')
	measures = []
	for r in r_list:
		nmf_r = NMF(n_components=r, init='random', random_state = 42).fit_transform(data_tfidf)
		pred_ = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(nmf_r)
		x,y = (print_5_measure(d.target,pred_))
		measures.append(x)
		print (x,y)
	measures = np.array(measures)
	for i in range(5):
		plt.plot(r_list,measures[:,i])
		plt.show()
	'''
	print ('_____4a_____')
	#svd_r = data_svd[:,:2]
	svd_r = TruncatedSVD(n_components=2, algorithm='arpack', random_state=42).fit_transform(data_tfidf)
	pred_ = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(svd_r)
	#class0 = np.array([j.tolist() for i,j in zip(pred_,svd_r) if i==0])
	#class1 = np.array([j.tolist() for i,j in zip(pred_,svd_r) if i==1])
	class0 = np.array([j.tolist() for i,j in zip(d.target,svd_r) if i==0])
	class1 = np.array([j.tolist() for i,j in zip(d.target,svd_r) if i==1])
	class0 = [class0[:,0],class0[:,1]]
	class1 = [class1[:,0],class1[:,1]]
	print (class0)
	
	data = (class0,class1)
	colors = ("red", "green")
	groups = ('comp','rec')

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
 
	for data, color, group in zip(data, colors, groups):
		x, y = data
		ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
	
	plt.title('SVD_2D')
	plt.legend(loc=2)
	plt.show()


def proprocess(data):
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	data_re = [regex.sub('', text) for text in data]

	tfidf = TfidfVectorizer(min_df=3,stop_words='english')
	data_tfidf = tfidf.fit_transform(data_re)
	return data_tfidf

def print_5_measure(y_true,y_pred):
	x1 = metrics.homogeneity_score(y_true,y_pred)
	x2 = metrics.completeness_score(y_true,y_pred)
	x3 = metrics.v_measure_score(y_true,y_pred)
	x4 = metrics.adjusted_rand_score(y_true,y_pred)
	x5 = metrics.adjusted_mutual_info_score(y_true,y_pred)
	x6 = contingency_matrix(y_true,y_pred)
	return [x1,x2,x3,x4,x5],x6

def main():
	cat8 = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
	cat20 = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian', 'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
	
	PART(cat8)
	PART(cat20)

	


if __name__ == '__main__':
	main()