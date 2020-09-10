# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:33:06 2020

@author: Sayan Mondal
"""
# Importing All Necessary Libraries ##
import numpy as np   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  

##.................Importing train & tst data................############
dataset= pd.read_csv("C:/Users/Sayan Mondal/Desktop/medicine-side-effects-analysis/train.csv")
data= pd.read_csv("C:/Users/Sayan Mondal/Desktop/medicine-side-effects-analysis/test.csv")


## FINDING NULL VALUES...##
dataset.isnull().sum() ## there is 219 null values in condition column which is 0.005% of entire value#
data.isnull().sum()

## droping all NA values from Train  data..##
dataset.dropna(inplace=True)

#############......Pre-processing on train data...###########################

dataset.describe()

dataset['rating'].value_counts()

##removing the na values from train data##
dataset.dropna(inplace=True)

## checking percentage wise distribution of rating..##
dataset['rating'].value_counts(normalize=True)*100

dataset['rating'].value_counts()
dataset['output'].value_counts()
## plotting the ratings to get a clear picture of rating distribution..##     
sns.distplot(dataset['rating'] ,hist=True, bins=100)

dataset['usefulCount'].max()
dataset['usefulCount'].min()

## Finding the Correlation Matrix...##
corrmat = dataset.corr() 
print(corrmat)

# removing the date column as date has not that significance in output##
dataset.drop(["date"],axis=1,inplace=True)

dataset.head()

## cleaning the data..##
## Cleaning the text input for betting understanding of Machine..##

##Converting all review into Lowercase..###
dataset['review']= dataset['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))

## removing punctuation from review..#
import string
dataset['review']=dataset['review'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))
                                                 

## Remove Numbers from review...##
dataset['review']=dataset['review'].str.replace('[0-9]','')


## removing all stopwords(english)....###
from nltk.corpus import stopwords

stop_words=stopwords.words('english')

dataset['review']=dataset['review'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

dataset.head(2)

# Lemmatization
from textblob import Word
dataset['review']= dataset['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


## Removing specific words which dont have much significance and higher frequency..##
n_req= ['one','first','effect','side','taking','day', 'month','year','week','im','ive','mg','time','hour','could','lb','two','sideeffect','started','still']

dataset['review']=dataset['review'].apply(lambda x: " ".join(word for word in x.split() if word not in n_req))




## subjectvity & polarity of each review rows...##
from textblob import TextBlob

dataset['polarity'] = dataset['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
dataset['subjectivity'] = dataset['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

## Finding sentiment through VADER sentiment Analyzer..##
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(dataset.iloc[4]['review'])

dataset['vad_scores'] = dataset['review'].apply(lambda review:sid.polarity_scores(review))
dataset['vad_compound'] = dataset['vad_scores'].apply(lambda d:d['compound'])


########...finding Correlation in the data....###
corrmat = dataset.corr() 
print(corrmat)


## ....Finding most common occuring words in Corpus...##
review_str=" ".join(dataset.review)
text=review_str.split()

from collections import Counter
counter= Counter(text)
top_100= counter.most_common(100)
print(top_100)

###############.....Finding Unique Words from the entire corpus...##################
len(set(counter))


###### WordCloud formation for better understanding of the data...##

from wordcloud import WordCloud
from PIL import Image
apple = np.array(Image.open( "C:/Users/Sayan Mondal/Desktop/apple.jpg"))

wordcloud= WordCloud(width= 3000,
                     height=3000,mask=apple,
                     background_color='black'
                     ).generate(review_str)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#######...Dividing the dataset into Good revies & Bad reviews..####################
dataset['review_type']= dataset['rating'].apply(lambda x: 'No' if x>=7 else 'Yes')
sns.countplot(dataset.review_type)
dataset['review_type'].value_counts(normalize=True)*100

#############....Good & Bad reviews WordCloud formation ################################

good_reviews= dataset[dataset.review_type=='Yes']
bad_reviews= dataset[dataset.review_type=='No']

good_reviews_text=" ".join(good_reviews.review.to_numpy().tolist())
bad_reviews_text=" ".join(bad_reviews.review.to_numpy().tolist())

good_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(good_reviews_text)
bad_reviews_cloud=WordCloud(background_color='black',max_words=100).generate(bad_reviews_text)

plt.imshow(good_reviews_cloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()

plt.imshow(bad_reviews_cloud,interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()


####...part2.....###### CLEANING OF TESTDATA .....########################################

##Lowercase 
data['review']= data['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))

## removing punctuation..#
data['review']=data['review'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))
                                                 

## Remove Numbers
data['review']=data['review'].str.replace('[0-9]','')


## removing stopwords
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
data['review']=data['review'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

dataset.head(2)

# Lemmatization
from textblob import Word
data['review']= data['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


data['polarity'] = data['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['subjectivity'] = data['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


## Finding sentiment through VADER sentiment Analyzer..##
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
data['vad_scores'] = data['review'].apply(lambda review:sid.polarity_scores(review))
data['vad_compound'] = data['vad_scores'].apply(lambda d:d['compound'])





###################.... Creating a NN Model...############################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
cv = TfidfVectorizer()
le = LabelEncoder() 

## convering into different levels..##
dataset['drugName']= le.fit_transform(dataset['drugName']) 
dataset['condition']= le.fit_transform(dataset['condition']) 
dataset['output']= le.fit_transform(dataset['output'])

X= dataset[['vad_compound','polarity','subjectivity']]
Y=dataset['output']

X_train, X_test, y_train, y_test = train_test_split(X, Y)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30,30))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train)*100
np.mean(y_test==prediction_test)*100

print("Accuracy:",metrics.accuracy_score(y_test, prediction_test ))

## ACCURACY=69.44%....##



#### appling SVM Model....###

from sklearn import svm
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report

X= dataset[['polarity','subjectivity','vad_compound']]
Y=dataset['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_test.shape
y_test.shape
classifier=svm.SVC(kernel='linear',gamma='auto', C=2)
model=classifier.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## with input polarity & subjectivity accuracy is 67.62% and with polarity and review_type accuracy is 45.54%..##


## Using Naive Bays Classifier...##
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


X= dataset.review
y=dataset.output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])

model=nb.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## accuracy =72.00%...


## Using Logistic regression Model ..###

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

X=dataset.review
y=dataset.output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1.0)),
               ])

model=logreg.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
 #### Accuracy of the model is 79%...###

print(classification_report(y_test,y_pred))
# confusion Matrix...##
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

## ROC Curve...

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Review Classification')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show() 


#######.... using oversampling to reduce class imbalance..#####################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



## Logistic Regression after using SMOTE ...##

tv = TfidfVectorizer(stop_words=None, max_features=34000)
X= tv.fit_transform(dataset.review)
y=dataset['output']

sm = SMOTE(random_state=444)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =122) 
X_train.shape
y_train.shape

X_train_res, y_train_res = sm.fit_resample(X_train, y_train) ## data increased by 62,025..#
X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape

print('Resampled dataset shape %s' % Counter(y_train_res))

##Resampled dataset shape Counter({1: 21252, 0: 21252})...###

model = LogisticRegression()
model.fit(X_train_res,y_train_res)

y_pred = model.predict(X_test)
count_misclassified = (y_test != y_pred).sum()

accuracy = metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred)) 

pd.crosstab(y_test, y_pred)
##accuracy is 78%...




########## SMOTE USING REVIEW COLUMN- LogisticRegression ...###################
tv = TfidfVectorizer(stop_words=None, max_features=34000)
X= tv.fit_transform(dataset.review)
y=dataset['output']


## spliting the data into 80:20 ratio...###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

smt = SMOTE(random_state=777)
X_train, y_train= smt.fit_sample(X_train, y_train)

X_train.shape
y_train.shape

print('Resampled dataset shape %s' % Counter(y))

logreg=LogisticRegression(solver = 'lbfgs')

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## acuracy of the model 79%...##

print(classification_report(y_test,y_pred))

pd.crosstab(y_test, y_pred)


########## SMOTE USING REVIEW COLUMN- RandomForest Classifier...###################
from sklearn.ensemble import RandomForestClassifier

tv = TfidfVectorizer(stop_words=None, max_features=40000)

X= tv.fit_transform(dataset.review)
y=dataset.output

sm = SMOTE(random_state=444)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =122) 
X_train.shape
y_train.shape

X_train_res, y_train_res = sm.fit_resample(X_train, y_train) ## data increased by 62,025..#
X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape

print('Resampled dataset shape %s' % Counter(y_train_res))

rf= RandomForestClassifier(n_estimators=50)

model=rf.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test,y_pred))

pd.crosstab(y_test, y_pred)

df = pd.DataFrame({ 'Actual': y_test, 'Predicted': y_pred})
df1=df.head(10)

print(df1)




####### SMOTE WITH Dependent Variable review......##################################

from sklearn.ensemble import RandomForestClassifier

tv = TfidfVectorizer(stop_words=None, max_features=35000)
X= tv.fit_transform(dataset.review)
y=dataset['rating']


## spliting the data into 80:20 ratio...###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 888) 


smt = SMOTE(random_state=777)
X_train, y_train= smt.fit_sample(X_train, y_train)

X_train.shape
y_train.shape

print('Resampled dataset shape %s' % Counter(y_train))



rf= RandomForestClassifier(n_estimators=50)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test,y_pred))

pd.crosstab(y_test, y_pred)

df = pd.DataFrame({ 'Actual': y_test, 'Predicted': y_pred})
df1=df.head(10)

print(df1)






### Final model that has been selected is RandomForest. Because accuracy is pretty good as compare to other model..###
## RandomForest took almost 10-15mins to run the entire file....####



#############......Use of Test File on the finalised MODEL-Random Forest Classifier   ##############################
rf= RandomForestClassifier(n_estimators=50)

######  MODEL APPLICATION_ 1....####


tv = TfidfVectorizer(stop_words=None, max_features=19612)
X= tv.fit_transform(dataset.review)
y=dataset['output']


## spliting the data into 80:20 ratio...###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 888) 


smt = SMOTE(random_state=777)
X_train, y_train= smt.fit_sample(X_train, y_train)

X_train.shape
y_train.shape

print('Resampled dataset shape %s' % Counter(y_train))


## TFIDF FOR TEST DATA...##
tv = TfidfVectorizer(stop_words=None, max_features=19612)
x_test1= tv.fit_transform(data.review)

rf= RandomForestClassifier(n_estimators=50)

model=rf.fit(X_train, y_train)

y_pred = model.predict(x_test1)



df = pd.DataFrame({'Id':data['Id'] ,'output': y_pred})

final_outcome=df.to_csv("C:/Users/Sayan Mondal/Desktop/medicine-side-effects-analysis/submission.csv", index=False)


######  MODEL APPLICATION_ 2....####


tv = TfidfVectorizer(stop_words=None, max_features=19612)
X= tv.fit_transform(dataset.review)
y=dataset['output']


## spliting the data into 80:20 ratio...###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

sm = SMOTE(random_state=777)
X_train_res, y_train_res= sm.fit_resample(X_train, y_train)

X_train_res.shape
y_train_res.shape

print('Resampled dataset shape %s' % Counter(y_train_res))

logreg=LogisticRegression()

model=logreg.fit(X_train_res, y_train_res)

## TFIDF FOR TEST DATA...##
tv = TfidfVectorizer(stop_words=None, max_features=19612)
x_test1= tv.fit_transform(data.review)

y_pred = model.predict(x_test1)

df = pd.DataFrame({'Id':data['Id'] ,'output': y_pred})

final_outcome=df.to_csv("C:/Users/Sayan Mondal/Desktop/medicine-side-effects-analysis/submission11.csv", index=False)



































































