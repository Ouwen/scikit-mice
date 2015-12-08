from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import numpy as np

class MiceImputer(object):

	def __init__(self, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
		self.missing_values = missing_values
		self.strategy = strategy
		self.axis = axis
		self.verbose = verbose
		self.copy = copy
		self.imp = Imputer(missing_values=self.missing_values, strategy=self.strategy, axis= self.axis, verbose=self.verbose, copy=self.copy)

	def _seed_values(self, X):
		self.imp.fit(X)
		return self.imp.transform(X)

	def _get_mask(X, value_to_mask):
	    if value_to_mask == "NaN" or np.isnan(value_to_mask):
	        return np.isnan(X)
	    else:
	        return X == value_to_mask

	def _process(self, X, column, model_class):
		# Remove values that are in mask
		mask = np.array(self._get_mask(X)[:, column].T)[0]
		mask_indices = np.where(mask==True)[0]
		X_data = np.delete(X, mask_indices, 0)

		# Instantiate the model
		model = model_class()

		# Slice out the column to predict and delete the column.
		y_data = X[:, column]
		X_data = np.delete(X_data, column, 1)

		# Split training and test data
		X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

		# Fit the model
		model.fit(X_train, y_train)

		# Score the model
		scores = model.score(X_test, y_test)

		# Predict missing vars
		X_predict = np.delete(X, column, 1)
		y = model.predict(X_predict)

		# Replace values in X with their predictions
		predict_indices = np.where(mask==False)[0]
		np.put(X, predict_indicies, np.take(y, predict_indices))
	
		# Return model and scores
		return (model, scores)
	
	def transform(self, X, model_class=LinearRegression, iterations=10):
		X = np.matrix(X)
		mask = _get_mask(X, self.missing_values)
		seeded = self._seed_values(X)
		specs = np.zeros(iterations, len(X.T))

		for i in range(iterations):
			for c in range(len(X.T) - 1):
				specs[i][c] = self._process(X, c, model_class)
		
		# Return X matrix with imputed values
		return (X, specs)