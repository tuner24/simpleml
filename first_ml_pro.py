#
'''
http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
data source: https://archive.ics.uci.edu/ml/datasets/Iris

'''
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)  # convert tabular data into a DataFrame object

# ---------Summarize the Dateset---------
def summarize_dataset():
	print(dataset.shape)
	print(dataset.head(20))
	print(dataset.describe())
	print(dataset.groupby('class').size())

# ---------Data Visualization---------
def data_visualization():
	# box and whisker plots
	dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	plt.show()

	# histotrams
	dataset.hist()
	plt.show()

	# scatter plot matrix
	scatter_matrix(dataset)
	plt.show()

# ---------Evaluate Some Algorithms---------
def main():
	# Split-out validation dataset
	array = dataset.values  # return a array
	X = array[:, 0:4]  # get the priority four rows, it's still a 'numpy.ndarray' type
	Y = array[:, 4]    # get the fifth row
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(
		X, Y, test_size=validation_size, random_state=seed)

	# Test options and evaluation metric
	num_folds = 10
	num_instances = len(X_train)
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = [
		('LR', LogisticRegression()),
		('LDA', LinearDiscriminantAnalysis()),
		('KNN', KNeighborsClassifier()),
		('CART', DecisionTreeClassifier()),
		('NB', GaussianNB()),
		('SVM', SVC())
	]
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
		cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	# Compare Algorithms
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

# Make predictions on validation dataset
def make_predictions():
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))


if __name__ == '__main__':
	main()








