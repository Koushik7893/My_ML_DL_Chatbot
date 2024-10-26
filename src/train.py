from src.components.Data_Ingestion import Data
from src.pipelines.train_models import ModelIntializer
from src.pipelines.models_params import ParamsInit
from src.components.Data_Transformation import ColumnTransformWithSplit
from src.helper import datasets_path, save_pkl_file, load_pkl_file, compute_classification_results, compute_regression_results, all_results_json
import os
columntransformsplit = ColumnTransformWithSplit()
mod_int = ModelIntializer()
params_init = ParamsInit()

class ModelTraining:
    def __init__(self):
        self.datasets_info = datasets_path()
        self.results = {}
        self.type_results = {}
        
    def data_loading(self, typeoftraining ,dataset_path, target='Target'):
        self.data = Data(dataset_path, target, typeoftraining)
        self.x_train, self.x_test, self.y_train, self.y_test, self.preprocessor, self.y_encoder = self.data.dataprocessing(columntransformsplit)
        
    
    def TrainAllDatasets(self):
        for typetotrain in list(self.datasets_info.keys()):    ## Classification, regression, etc.
            dataset_accuracy = {}
            model_names = mod_int.get(typetotrain)    ## Get all Models names
            datasets = self.datasets_info[typetotrain]    ## Get all Datasets names, paths
            
            for dataset_name, dataset_path in datasets.items():    ## Datasets names, paths
                model_results = {}
                self.data_loading(typetotrain, dataset_path)    ## Load Data
                column_transformer_path = f'artifacts/{typetotrain}/{dataset_name}/{dataset_name}_preprocessor.pkl'
                label_encoder_path = f'artifacts/{typetotrain}/{dataset_name}/{dataset_name}_encoder.pkl'
                if not os.path.exists(column_transformer_path):
                        save_pkl_file(column_transformer_path, self.preprocessor)
                if not os.path.exists(label_encoder_path) and self.y_encoder != None:
                        save_pkl_file(label_encoder_path, self.y_encoder)
                
                for model_name in model_names:
                    model_path = f'artifacts/{typetotrain}/{dataset_name}/models/{model_name}.pkl'
                    if not os.path.exists(model_path):                       
                        model = mod_int.model_intializer(model_name, typetotrain) 
                        cv = params_init.Grid_SearchCV(typetotrain, model(), model_name)
                        cv.fit(self.x_train, self.y_train)
                        best_params, best_model = cv.best_params_, cv.best_estimator_
                        save_pkl_file(model_path, best_model)
                        y_pred = best_model.predict(self.x_test)
                        if typetotrain == 'classification':
                            model_res = compute_classification_results(y_pred, self.y_test)
                        else:
                            model_res = compute_regression_results(y_pred, self.y_test)
                        model_results[model_name] = {'best_parms':best_params,**model_res}   
                    print(model_name)
                print(dataset_name)
                if model_results:                       
                    dataset_accuracy[dataset_name] = model_results
            print(typetotrain)
            if dataset_accuracy:
                self.results[typetotrain] = dataset_accuracy
                
        return all_results_json(self.results)
    
    
    
