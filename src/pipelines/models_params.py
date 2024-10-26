from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
import numpy as np

available_params = {
    'LogisticRegression': {
        'penalty': 'str, default="l2". Regularization type (l1, l2, elasticnet).',
        'dual': 'bool, default=False. Dual or primal formulation.',
        'C': 'float, default=1.0. Inverse of regularization strength.',
        'fit_intercept': 'bool, default=True. Whether to calculate the intercept.',
        'solver': 'str, default="lbfgs". Algorithm to use in the optimization problem.',
        'max_iter': 'int, default=100. Maximum number of iterations.',
        'n_jobs': 'int, default=None. Number of parallel jobs to run.'
    },
    'SVC': {
        'C': 'float, default=1.0. Regularization parameter.',
        'kernel': 'str, default="rbf". Kernel type (linear, poly, rbf, sigmoid).',
        'degree': 'int, default=3. Degree of the polynomial kernel function.',
        'gamma': 'str or float, default="scale". Kernel coefficient.',
        'probability': 'bool, default=False. Whether to enable probability estimates.',
    },
    'GaussianNB': {
        'var_smoothing': 'float, default=1e-9. Portion of the largest variance of all features added to variances.',
    },
    'KNeighborsClassifier': {
        'n_neighbors': 'int, default=5. Number of neighbors to use.',
        'weights': 'str, default="uniform". Weight function ("uniform", "distance").',
        'algorithm': 'str, default="auto". Algorithm used to compute nearest neighbors.',
        'leaf_size': 'int, default=30. Leaf size for BallTree or KDTree.',
    },
    'DecisionTreeClassifier': {
        'criterion': 'str, default="gini". Function to measure split quality ("gini", "entropy").',
        'splitter': 'str, default="best". Strategy to split nodes ("best", "random").',
        'max_depth': 'int, default=None. Maximum depth of the tree.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
    },
    'RandomForestClassifier': {
        'n_estimators': 'int, default=100. Number of trees in the forest.',
        'criterion': 'str, default="gini". Function to measure split quality.',
        'max_depth': 'int, default=None. Maximum depth of the trees.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
        'n_jobs': 'int, default=None. Number of parallel jobs to run.',
    },
    'GradientBoostingClassifier': {
        'learning_rate': 'float, default=0.1. Learning rate shrinks the contribution of each tree.',
        'n_estimators': 'int, default=100. Number of boosting stages.',
        'max_depth': 'int, default=3. Maximum depth of individual regression estimators.',
        'subsample': 'float, default=1.0. Fraction of samples used for fitting the individual base learners.',
    },
    'AdaBoostClassifier': {
        'n_estimators': 'int, default=50. Number of weak learners (trees).',
        'learning_rate': 'float, default=1.0. Weight applied to each classifier.',
    },
    'XGBClassifier': {
        'n_estimators': 'int, default=100. Number of boosting rounds.',
        'max_depth': 'int, default=6. Maximum depth of a tree.',
        'learning_rate': 'float, default=0.3. Boosting learning rate.',
        'subsample': 'float, default=1.0. Fraction of samples used for training.',
    },
    'LinearRegression': {
            'fit_intercept': 'bool, default=True. Whether to calculate the intercept.',
            'normalize': 'bool, default=False. Deprecated; used to normalize before regression.',
            'copy_X': 'bool, default=True. If True, X will be copied; else, overwritten.',
            'n_jobs': 'int, default=None. Number of parallel jobs to run.'
    },
    'Lasso': {
        'alpha': 'float, default=1.0. Constant that multiplies the L1 term.',
        'max_iter': 'int, default=1000. Maximum number of iterations.',
        'tol': 'float, default=0.0001. Tolerance for the optimization.',
    },
    'Ridge': {
        'alpha': 'float, default=1.0. Regularization strength.',
        'solver': 'str, default="auto". Solver to use in the optimization problem.',
    },
    'ElasticNet': {
        'alpha': 'float, default=1.0. Constant that multiplies the penalty terms.',
        'l1_ratio': 'float, default=0.5. The mix ratio between L1 and L2 regularization.',
        'max_iter': 'int, default=1000. Maximum number of iterations.',
    },
    'SVR': {
        'kernel': 'str, default="rbf". Specifies the kernel type to be used.',
        'degree': 'int, default=3. Degree of the polynomial kernel.',
        'C': 'float, default=1.0. Regularization parameter.',
        'epsilon': 'float, default=0.1. Epsilon-tube within which no penalty is given.',
    },
    'KNeighborsRegressor': {
        'n_neighbors': 'int, default=5. Number of neighbors to use.',
        'weights': 'str, default="uniform". Weight function ("uniform", "distance").',
        'algorithm': 'str, default="auto". Algorithm used to compute nearest neighbors.',
    },
    'DecisionTreeRegressor': {
        'criterion': 'str, default="mse". Function to measure split quality ("mse", "mae").',
        'splitter': 'str, default="best". Strategy to split nodes.',
        'max_depth': 'int, default=None. Maximum depth of the tree.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
    },
    'RandomForestRegressor': {
        'n_estimators': 'int, default=100. Number of trees in the forest.',
        'criterion': 'str, default="mse". Function to measure split quality.',
        'max_depth': 'int, default=None. Maximum depth of the trees.',
        'min_samples_split': 'int, default=2. Minimum number of samples required to split an internal node.',
        'n_jobs': 'int, default=None. Number of parallel jobs to run.'
    },
    'GradientBoostingRegressor': {
        'learning_rate': 'float, default=0.1. Shrinks contribution of each tree.',
        'n_estimators': 'int, default=100. Number of boosting stages.',
        'max_depth': 'int, default=3. Maximum depth of the individual estimators.',
        'subsample': 'float, default=1.0. Fraction of samples used for fitting the individual base learners.',
    },
    'AdaBoostRegressor': {
        'n_estimators': 'int, default=50. Number of weak learners.',
        'learning_rate': 'float, default=1.0. Weight applied to each regressor.',
    },
    'XGBRegressor': {
        'n_estimators': 'int, default=100. Number of boosting rounds.',
        'max_depth': 'int, default=6. Maximum depth of a tree.',
        'learning_rate': 'float, default=0.3. Boosting learning rate.',
        'subsample': 'float, default=1.0. Fraction of samples used for training.',
    }
}
params = {
        'LogisticRegression': {
            'penalty': ['l1', 'l2', 'elasticnet'], 
            'C': [0.1, 0.5, 1.0, 10.0], 
            'fit_intercept': [True, False],            
            'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'], 
            'max_iter': [100, 200, 300, 500],  
            'n_jobs': [-1]
        },
        'SVC': {
            'C': [0.1, 0.5, 1.0, 10.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
            'probability': [True, False]
        },
        'GaussianNB': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40, 50]
        },
        'DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15,30],
            'min_samples_split': [2, 5, 10]
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'n_jobs': [-1]
        },
        'GradientBoostingClassifier': {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'subsample': [0.5, 0.75, 1.0]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.5, 1.0, 1.5]
        },
        'XGBClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.75, 1.0]
    },
    
        'LinearRegression': {
            'fit_intercept': [True, False],
            'n_jobs': [-1]
        },
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0],
            'max_iter': [100, 500, 1000],
            'tol': [0.0001, 0.001, 0.01]
        },
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
            'tol': [0.0001, 0.001, 0.01]
        },
        'ElasticNet': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [100, 500, 1000]
        },
        'SVR': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 10],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.5]
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'DecisionTreeRegressor': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'n_jobs': [-1]
        },
        'GradientBoostingRegressor': {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'subsample': [0.5, 0.75, 1.0]
        },
        'AdaBoostRegressor': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.5, 1.0, 1.5]
        },
        'XGBRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.75, 1.0]
    }
}


class ParamsInit:
    def __init__(self):
        self.params = params
    
    def Grid_SearchCV(self,typetotrain, model, model_name):
        model_params = self.params[model_name]
        if typetotrain == "classification":
            search = GridSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'accuracy')
        else:
            search = GridSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'neg_mean_squared_error')
        return search
    
    def Randomized_SearchCV(self,typetotrain, model, model_name):
        model_params = self.params[model_name]
        count = 0
        for i in model_params.values():
            count += len(i)
        n_iter = np.round(count * 3 / 4)
        if typetotrain == "classification":
            search = RandomizedSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'accuracy', n_iter=n_iter)
        else:
            search = RandomizedSearchCV(estimator=model, param_grid=model_params, cv=5, scoring = 'neg_mean_squared_error', n_iter=n_iter)
        return search