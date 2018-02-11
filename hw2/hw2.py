from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
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
from sklearn.preprocessing import StandardScaler
import math
from sklearn.cluster import KMeans


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
	data_tfidf = preprocess(d.data)
	print (data_tfidf.shape)
	print ('_____2a_____')
	pred_ = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_tfidf)
	print ('_____2b_____')
	print (print_5_measure(d.target,pred_))
	print ('_____3a_i_____')
	'''
	svd = TruncatedSVD(n_components=1000, algorithm='arpack', random_state=42)
	data_svd = svd.fit_transform(data_tfidf)
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
	'''
	print ('_____3a_ii_____')
	measures_list = ['homogeneity','completeness','v-measure','rand score','mutual information']
	print ('_____SVD_____')
	svd = TruncatedSVD(n_components=300, algorithm='arpack', random_state=42)
	data_svd = svd.fit_transform(data_tfidf)
	r_list = [1,2,3,5,10,20,50,100,300]
	#r_list = np.arange(2,50)
	measures = []
	for r in r_list:
		svd_r = data_svd[:,:r]
		pred_ = KMeans(n_clusters=cats, random_state=0).fit_predict(svd_r)
		x,y = (print_5_measure(d.target,pred_))
		measures.append(x)
		print (x,y)
	measures = np.array(measures)
	best_r_svd = r_list[np.argmax(measures[:,0])]
	print (best_r_svd)
	for i in range(5):
		plt.plot(r_list,measures[:,i],label=measures_list[i])
	plt.legend(loc=1)
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
	best_r_nmf = r_list[np.argmax(measures[:,0])]
	for i in range(5):
		plt.plot(r_list,measures[:,i],label=measures_list[i])
	plt.legend(loc=1)
	plt.show()
	
	print ('_____4a_____')
	svd = TruncatedSVD(n_components=best_r_svd, algorithm='arpack', random_state=42).fit_transform(data_tfidf)
	nmf = NMF(n_components=best_r_nmf, init='random', random_state = 42).fit_transform(data_tfidf)

	visualize(svd,d.target,cats,nmf=False)
	visualize(nmf,d.target,cats,nmf=True)
	print ('_____4a_ii_____')
	print_label('1')
	data_svd = TruncatedSVD(n_components=best_r_svd, algorithm='arpack', random_state=42).fit_transform(data_tfidf)
	data_norm = StandardScaler(with_mean=False).fit_transform(data_svd)
	pred_svd = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_norm)
	visualize(data_norm,d.target,cats,nmf=False)
	print(print_5_measure(d.target,pred_svd))

	data_nmf = NMF(n_components=best_r_nmf, init='random', random_state = 42).fit_transform(data_tfidf)
	data_norm = StandardScaler(with_mean=False).fit_transform(data_nmf)
	pred_nmf = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_norm)
	visualize(data_norm,d.target,cats,nmf=True)
	print(print_5_measure(d.target,pred_nmf))
	
	print_label('2')
	data_nmf = NMF(n_components=best_r_nmf, init='random', random_state = 42).fit_transform(data_tfidf)
	data_log = np.log(data_nmf+0.001)
	pred_nmf = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_log)
	visualize(data_log,d.target,cats,nmf=True)
	print(print_5_measure(d.target,pred_nmf))

	print_label('log->nprm')
	data_log_norm = StandardScaler(with_mean=False).fit_transform(data_log)
	pred_nmf = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_log_norm)
	visualize(data_log_norm,d.target,cats,nmf=True)
	print(print_5_measure(d.target,pred_nmf))

	print_label('norm->log')
	data_norm_log = np.log(data_norm+0.001)
	pred_nmf = MiniBatchKMeans(n_clusters=cats, init='k-means++', n_init=30, random_state=42).fit_predict(data_norm_log)
	visualize(data_norm_log,d.target,cats,nmf=True)
	print(print_5_measure(d.target,pred_nmf))


def print_label(i):
	print ('---------------')
	print ('       '+i)
	print ('---------------')
def visualize(vector,target,C,nmf):
	if nmf:
		vector = TruncatedSVD(n_components=2,algorithm='arpack',random_state=42).fit_transform(vector)
	class_ = []
	for c in range(C):
		tmp = np.array([j.tolist() for i,j in zip(target,vector) if i==c])
		class_.append([tmp[:,i] for i in range(2)])
	data = class_
	colors = [np.random.rand(3,) for i in range(C)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
 
	for data, color in zip(data, colors):
		x, y = data
		ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
	
	if nmf:
		plt.title('NMF_2D')
	else:
		plt.title('SVD_2D')
	plt.show()

def preprocess(data):
	stop_words_skt = text.ENGLISH_STOP_WORDS
	stop_words_en = stopwords.words('english')
	combined_stopwords = set.union(set(stop_words_en),set(string.punctuation),set(stop_words_skt))
	data_tfidf = TfidfVectorizer(min_df=3,stop_words=combined_stopwords).fit_transform(data)

	#regex = re.compile('[%s]' % re.escape(string.punctuation))
	#data_re = [regex.sub('', text) for text in data]

	#data_tfidf = TfidfVectorizer(min_df=3,stop_words=stop_words_skt).fit_transform(data_re)
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