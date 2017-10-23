"""
A simple script that demonstrates how we classify textual data with sklearn.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
#from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn import tree
#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:

        review,rating=line.strip().split('\t')
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
#00000
    return reviews,labels
def final(i):
    if i[0]==i[1]
	
rev_train,labels_train=loadData('training.txt')
f=open('testing.txt')
for line in f:
    review=line.strip()
    rev_test.append(review.lower())
f.close
rev_train,labels_train=loadData('training.txt')
#rev_train = re.findall(r'\w+', rev_train,flags = re.UNICODE | re.LOCALE)
#Build a counter based on the training dataset
counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(rev_train)
#counter.fit(rev_train)
#stop_words=stopwords.words('english')
#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

NB = MultinomialNB()
NB.fit(counts_train,labels_train)

LR = LogisticRegression(C=0.1,max_iter=1000)
LR.fit(counts_train,labels_train)

SGD = SGDClassifier()
SGD.fit(counts_train,labels_train)

SVC = LinearSVC()
SVC.fit(counts_train,labels_train)

DT = tree.DecisionTreeClassifier()
DT.fit(counts_train,labels_train)


#use the classifier to predict
predicted_NB=NB.predict(counts_test)
predicted_LR=LR.predict(counts_test)
predicted_SGD=SGD.predict(counts_test)
predicted_SVC=SVC.predict(counts_test)
predicted_DT=DT.predict(counts_test)


abc=zip(predicted_NB,predicted_SGD,predicted_LR,predicted_SVC,predicted_DT)

results=[]
for i in abc:
	results.append(final(i))

fw = open('out.txt','w')
content = ''
for result in results:
    content += str(result) + '\n'
fw.write(content)
fw.close()


print accuracy_score(results,labels_test)

