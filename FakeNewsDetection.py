import itertools
import cPickle as c
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request

#Reading the dataset using pandas
df = pd.read_csv('fake_or_real_news.csv')
df.set_index('Unnamed: 0', inplace=True)

#Extracting the Y(Label) data
y = df['label']
df.drop('label', axis=1, inplace=True)

#Splitting the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

#Count Vector initialization
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(x_train)
count_test = count_vectorizer.transform(x_test)

#Training the classifier
clf = MultinomialNB()
clf.fit(count_train, y_train)

#Initializing the flask server
app = Flask(__name__)

#Initializing the base page
@app.route('/')
def index():
	return render_template("FakeNewsDetection.html")

#Initializing the about page
@app.route('/About.html')
def about():
	return render_template("About.html")

#Initializing the predict page
@app.route('/predict.html',methods = ['POST','GET'])
def predict():
	#Getting the data from the fields
	if request.method == 'POST':
			newstitle = request.form["newstitle"]
			newsbody = request.form["newsbody"]
	
	#Converting the inputs into series
	data_series = pd.Series([newsbody])

	#Performing Count Vectorizer transformation 
	new_count_test = count_vectorizer.transform(data_series)

	#Predicting the output based on the given input
	count_predicted_value = clf.predict(new_count_test)

	#Confusion Matrix plot
	def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
		#Setting matplotlib configuration
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(os.path.join('static','fakenews.png'))
		plt.gcf().clear()

	#Calculating the values for the confusion matrix
	pred = clf.predict(count_test)
	score = metrics.accuracy_score(y_test, pred)
	cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
	plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

	#Passing control over predict html webpage with arguments
	return render_template("predict.html",newstitle=newstitle,newsbody=newsbody,predicted=count_predicted_value[0])

	
#Hosting the server with debugger configuration
if __name__ == '__main__':
	app.run(debug = True,host='0.0.0.0',port=5000)
