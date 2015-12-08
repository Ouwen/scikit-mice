# Scikit-mice

Scikit-mice runs the MICE imputation algorithm. Based on the following <a href="http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/">paper</a>.


### Documentation:
The MiceImputer class is similar to the sklearn <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html">Imputer</a> class. 

MiceImputer has the same instantiation parameters as Imputer.

The MiceImputer.transform() function takes in three arguments.

| Param                 | Type         | Description                                      |
| --------------------- | ------------ | ------------------------------------------------ |
| `X`                   | `matrix`     | Numpy matrix or python matrix of data.           |
| `model_class`         | `class`      | Scikit-learn model class.                        |
| `iterations`          | `int`        | Int for numbe of interations to run.             |


What is returned by MiceImputer is a tuple of imputed values as well as a matrix of model performance for each iteration and column.
```
(imputed_x, model_specs_matrix)
```

### Example:

```
from sklearn.linear_model import LinearRegression
import skmice

imputer = MiceImputer()
X = [[1, 2], [np.nan, 3], [7, 6]]

X, specs = imputer.transform(X, LinearRegression, 10)

print specs

```

What is returned is a MICE imputed matrix running 10 iterations using a simple LinearRegression.