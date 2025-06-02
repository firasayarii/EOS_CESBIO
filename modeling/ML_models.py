from warnings import filterwarnings
filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,ShuffleSplit,GridSearchCV,KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error
from math import sqrt
import joblib
df=pd.read_csv('preprocessed_data.csv')
X=df.drop(columns=['alpha_exp'])
y=df['alpha_exp']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
models = {
    'Random Forest': RandomForestRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Support Vector Regression': SVR()
}
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Extra Trees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    },
    'CatBoost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'depth': [4, 6, 8]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9 , 10 , 12],
        'weights': ['uniform', 'distance']
    },
    'Support Vector Regression': {
        'kernel': ['rbf', 'poly'],
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto']
    }
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
grids = {}
for model_name, model in models.items():
    grids[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grids[model_name].fit(X_train, y_train)
    best_params =  grids[model_name].best_params_
    best_score =(-1 * grids[model_name].best_score_)
    results.append({'Model': model_name, 'Best Score': best_score})
# Créer un DataFrame à partir des résultats
results_df = pd.DataFrame(results)
results_df.to_csv('modeling_results.csv',index=False)
metrics_results = []
for model_name, model in models.items():
    # Récupérer le meilleur modèle après GridSearchCV
    best_model = grids[model_name].best_estimator_
    
    # Prédire les valeurs sur X_test
    y_pred = best_model.predict(X_test)
    
    # Calculer le MSE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Ajouter le résultat au dictionnaire
    metrics_results.append({'Model': model_name, 'MSE': mse, 'MAE': mae})
# Créer un DataFrame avec les résultats MSE
metrics_results = pd.DataFrame(metrics_results)
metrics_results.to_csv('metrics_results.csv', index=False)
best_result = min(results, key=lambda x: x['Best Score'])
best_model_name = best_result['Model']
best_params = grids[best_model_name].best_params_
print(f"Meilleurs hyperparamètres pour {best_model_name} :", best_params)
best_model = grids[best_model_name].best_estimator_
joblib.dump(best_model, 'model1.pkl')
