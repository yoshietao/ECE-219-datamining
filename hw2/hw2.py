from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn import metrics


class Data:
	def __init__(self,cat8,cat20):
		data1_train = fetch_20newsgroups(subset='train', categories=cat8, random_state=42)
		data2_train = fetch_20newsgroups(subset='train', categories=cat20, random_state=42)
		self.train_data1 = data1_train.data
		self.train_target1 = np.array([int(i>3) for i in data1_train.target])
		self.train_data2 = data2_train.data
		self.train_target2 = data2_train.target
		

def proprocess(data):
	stop_words = text.ENGLISH_STOP_WORDS
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	data_re = [regex.sub('', text) for text in data]

	tfidf = TfidfVectorizer(min_df=3,stop_words=stop_words)
	data_tfidf = tfidf.fit_transform(data_re)
	return data_tfidf


def main():
	cat8 = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
	cat_all = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian', 'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
	
	d = Data(cat8,cat_all)
	print ('_____1_____')
	data_tfidf = proprocess(d.train_data1)
	print (data_tfidf.shape)
	print ('_____2a_____')
	kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(data_tfidf)
	pred_  = kmeans.predict(data_tfidf)
	cont_  = contingency_matrix(d.train_target1,pred_)
	print (cont_)
	print ('----_2b----_')
	print (metrics.homogeneity_completeness_v_measure(d.train_target1,pred_))
	print (metrics.adjusted_rand_score(d.train_target1,pred_))
	print (metrics.adjusted_mutual_info_score(d.train_target1,pred_))


if __name__ == '__main__':
	main()