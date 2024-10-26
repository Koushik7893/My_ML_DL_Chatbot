import os
from ..helper import json_as
# python -m src.pipelines.k
l = {}
data = {
    'classification':{
        'iris':{'Linear':{'best_params':{},**l},
                'hj':{'best_params':{},**l},},
        'p':{'Linxcear':{'best_params':{},**l},
                'hcbxj':{'best_params':{},**l},}
        },
    'regression':{
        'iris':{'Linear':{'best_params':{},**l},
                'hj':{'best_params':{},**l},},
        'p':{'Linxcear':{'best_params':{},**l},
                'hcbxj':{'best_params':{},**l},}
    }
    }


if __name__ == '__main__':
    json_as('artifacts/all_results.json', data)