import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Reading the dataset using pandas
df = pd.read_csv('fake_or_real_news.csv')
df.set_index('Unnamed: 0', inplace=True)

#Extracting the Y(Label) data
y = df['label']
df.drop('label', axis=1, inplace=True)

#Splitting the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=52)

#Count Vector initialization
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(x_train)
count_test = count_vectorizer.transform(x_test)

#Training the classifier
clf = MultinomialNB()
clf.fit(count_train, y_train)

#Predicting
predicted = clf.predict(count_test)
print(accuracy_score(predicted, y_test))

#Finding Coefficients
coef = clf.coef_.ravel()
top_negative_coef_index = np.argsort(coef)[:20]
top_positive_coef_index = np.argsort(coef)[-20:]
top_coef = np.hstack([top_negative_coef_index, top_positive_coef_index])
coef[top_positive_coef_index] = -1 * coef[top_positive_coef_index]

#Plotting
plt.figure(figsize=(15, 5))
colors = ['red' if c < -6 else 'blue' for c in coef[top_coef]]
plt.bar(np.arange(2 * 20), coef[top_coef], color=colors)
feature_names = np.array(count_vectorizer.get_feature_names())
plt.xticks(np.arange(1, 1 + 2 * 20), feature_names[top_coef], rotation=60, ha='right')
Axes.set_ylim(0, -6)
plt.show()
