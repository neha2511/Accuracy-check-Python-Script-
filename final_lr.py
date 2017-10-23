from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords


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
    return reviews,labels

rev_test=[]

f=open('testing.txt')
for line in f:
    review=line.strip()
    rev_test.append(review.lower())
f.close
rev_train,labels_train=loadData('training.txt')

#Build a counter based on the training dataset
counter = CountVectorizer()
counter.fit(rev_train)

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

#build a Logistic Regression classifier on the training data
LR=MultinomialNB(alpha=0.216)
LR.fit(counts_train,labels_train)
#use the classifier to predict
predicted_lr=LR.predict(counts_test)
fw=open('out.txt','w')

for i in predicted_lr:
	fw.write(str(i)+'\n')
fw.close()

