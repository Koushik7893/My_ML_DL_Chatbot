import os
import pickle
# import dill
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def datasets_path():
    data = {}
    types = os.listdir(path='data/preprocessed_data')
    for typeofclass in types:
        paths = {}
        datasets = os.listdir(path=f'data/preprocessed_data/{typeofclass}')
        for dataset in datasets:
            if dataset.endswith('.csv'):
                paths[str.title(os.path.splitext(dataset)[0])] = os.path.join(os.getcwd(),f'data/preprocessed_data/{typeofclass}/{dataset}')
        data[typeofclass] = paths
    return data

def save_pkl_file(file_path, model):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(model,file)
        
        
def load_pkl_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
    
def compute_classification_results(y_true, y_pred):
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'), 
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    #   'classification_report': classification_report(y_true, y_pred, output_dict=True),  
        'roc_auc': roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr') if len(set(y_true)) == 2 else 'N/A'
    }
    return results

def compute_regression_results(y_true, y_pred):
    results = {
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'root_mean_squared_error': root_mean_squared_error(y_true, y_pred),  
        'r2_score': r2_score(y_true, y_pred),
    }
    return results

def json_as(path, results):
    with open(path, "w") as file:
        json.dump(results, file)
        
def json_load(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def all_results_json(results):
    path = 'artifacts/all_results.json'
    if os.path.exists(path):
        loaded_results = json_load(path)
        for types, datasets in results.items():
            if types not in loaded_results.keys():
                loaded_results[types] = {}
            for dataset, models in datasets.items():
                if dataset not in loaded_results[types].keys():
                    loaded_results[types][dataset] = {}
                for model, model_results in models.items():
                    if model not in loaded_results[types][dataset].keys():
                        loaded_results[types][dataset][model] = model_results
        return json_as(path=path, results=loaded_results)
                
    else:
        return json_as(path=path, results=results)
    
    
    
    
# def dataset_info():
#     data = {}
#     types = os.listdir(path='data')
#     for typeofclass in types:
#         datasets = os.listdir(path=f'data/{typeofclass}')
#         sets = []
#         for dataset in datasets:
#             sets.append(dataset)
#         data[typeofclass] = sets
#     return data
    
    
# def flatten_results(results):
#     flattened_data = []
#     for typetotrain, data in results.items():
#         for dataset, models in data.items():
#             for model, metrics in models.items():
#                 for metric, value in metrics.items():
#                     flattened_data.append([typetotrain, dataset, model, metric, value])
#     df = pd.DataFrame(flattened_data, columns=['Type', 'Dataset', 'Model', 'Metric', 'Value'])
#     return df
