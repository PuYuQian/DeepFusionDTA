import h5py
import pickle
import numpy as np
import json
from collections import OrderedDict, defaultdict
import random as rn
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lightgbm import LGBMRegressor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
np.random.seed(1)
rn.seed(1)

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair is not 0:
        return summ/pair
    else:
        return 0

def read_data(train_Y_file, test_Y_file, feature_file):

    test_Y = json.load(open(test_Y_file))
    train_Y = json.load(open(train_Y_file))
    f = h5py.File(feature_file,'r')   #打开h5文件                   
    train_feature = f['train_feature'][:]
    test_feature = f['test_feature'][:]
    f.close()
    #scaler = MinMaxScaler()
    return train_Y, test_Y, train_feature, test_feature

def experiment(regressor, regressor_args):
    with open(f'log0918lgbbagging_n800.txt', 'w') as f:
        clf = regressor(**regressor_args)
        print(f'Present regressor is {str(regressor)}', file=f)
        #regr = BaggingRegressor(base_estimator=clf, n_estimators=10, random_state=1, n_jobs=-1, verbose=1)
        regr = BaggingRegressor(base_estimator=clf, n_estimators=800, random_state=1, n_jobs=10, verbose=1)
        regr.fit(train_feature, train_Y)  

        #best_estim.fit(train_feature, train_Y)
        print(f'params:{regr.get_params}', file=f)
        #pickle.dump(regr, open(f'best_bagging_0611_2.pkl', "wb"))
        #clf2 = pickle.load(open("best_boston.pkl", "rb"))
        gbdt_pred = regr.predict(test_feature)

        mse = mean_squared_error(test_Y, gbdt_pred)
        ci = get_cindex(test_Y, gbdt_pred)

        print(f'MSE:{mse},CI:{ci}', file=f)



train_Y, test_Y, train_feature, test_feature = read_data(train_Y_file="train_Y.txt", test_Y_file="test_Y.txt", feature_file='DenseFeature.h5')
regressor = (LGBMRegressor, )
regressor_args = defaultdict(dict,{LGBMRegressor: dict(random_state=1, max_depth=120, n_estimators=1200, num_leaves=50, learning_rate=0.01)})
#grid_args = defaultdict(dict,{LGBMRegressor: dict(max_depth=[160, 180, 200, 220], n_estimators=[1800, 2000, 2200], num_leaves=[80, 100, 120], learning_rate=[0.01,0.001])})
"""
regressor = (LGBMRegressor, XGBRegressor)
regressor_args = defaultdict(dict,{XGBRegressor: dict(random_state=1, learning_rate=0.001, ), LGBMRegressor: dict(random_state=1, metric='mse', bagging_fraction = 0.8, feature_fraction = 0.8)})
grid_args = defaultdict(dict,{XGBRegressor: dict(max_depth=[200],n_estimators=[2000,3000]), LGBMRegressor: dict(max_depth=[200, 250], n_estimators=[2000,3000], num_leaves=[60,100], learning_rate=[0.01,0.001])})
"""
for reg in regressor:
    args = (reg, regressor_args[reg])
    experiment(*args)