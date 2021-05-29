import numpy as np
import pandas as pd
import sklearn
import itertools
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import seaborn as sb
import pickle

#Read the data
df=pd.read_csv('fake_or_real_news.csv')
#Get shape and head
df.shape
df.head()

def create_distribution(dataFile):
    return sb.countplot(x='label', data=dataFile, palette='hls')

create_distribution(df)

def data_qualitycheck():
    print("checking data qualities...")
    df.isnull().sum()
    df.info()
    print("check finished")

data_qualitycheck()

y=df.label
y.head()
df.drop("label", axis=1)

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
X_train.head(10)
X_test.head(10)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)


def get_count_vectorizer_stats():
    print(count_train.shape)

get_count_vectorizer_stats()

count_test = count_vectorizer.transform(X_test)
tfidf_Vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_Vectorizer.fit_transform(X_train)

def get_tfidf_stats():
    tfidf_train.shape

get_tfidf_stats()

tfidf_test = tfidf_Vectorizer.transform(X_test)


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_Vectorizer.get_feature_names())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)


nb_pipeline = Pipeline([('NBTV',tfidf_Vectorizer), ('nb_clf', MultinomialNB())])
nb_pipeline.fit(X_train, y_train)
predicted_nbt = nb_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbt)
print(f'Accuracy: {round(score * 100, 2)}%')

nbc_pipeline = Pipeline([('NBCV',count_vectorizer), ('nb_clf', MultinomialNB())])
nbc_pipeline.fit(X_train, y_train)
predicted_nbc = nbc_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
print(f'Accuracy: {round(score * 100, 2)}%')

linear_clf = Pipeline([('linear', tfidf_Vectorizer), ('pa_clf', PassiveAggressiveClassifier(max_iter=50))])
linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print(f'Accuracy: {round(score * 100, 2)}%')

import pickle
model_file = 'model.pkl'
pickle.dump(linear_clf, open(model_file, 'wb'))
