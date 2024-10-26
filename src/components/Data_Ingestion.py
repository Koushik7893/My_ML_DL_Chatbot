import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class Data:
    def __init__(self, dataset, target, classtype):
        self.data = pd.read_csv(f'{dataset}')
        self.X = self.data.drop(columns=[target,'Unnamed: 0'])
        self.y = self.data[target].to_numpy()
        self.target = target
        self.classtype = classtype
        
    def info(self):
        return self.data.info()
    
    def columns_split(self):
        numeric_features = [feature for feature in self.X.columns if self.X[feature].dtype != "O"]
        categorical_features = [feature for feature in self.X.columns if self.X[feature].dtype == "O"]
        return numeric_features, categorical_features
    
    def labelencoding(self, cts):
        y_encoder = None
        y = self.y
        if self.y.dtype == "O":
            y_encoder = cts.label_encoder()
            y = y_encoder.fit_transform(self.y)
        return y_encoder, y
        
    def dataprocessing(self, cts):
        y_enc, self.y = self.labelencoding(cts)
        X_train, X_test, y_train, y_test = cts.get(self.X, self.y)
        numeric_features, categorical_features = self.columns_split()
        X_preprocessor = cts.column_transformer(numeric_features, categorical_features)
        X_train_transformed_data = X_preprocessor.fit_transform(X_train)
        X_test_transformed_data = X_preprocessor.transform(X_test)
        return X_train_transformed_data, X_test_transformed_data, y_train, y_test, X_preprocessor, y_enc
        
