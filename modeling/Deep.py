import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import json
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

df=pd.read_csv('preprocessed_data_.csv')
X=df.drop(columns=['alpha_exp'])
y=df['alpha_exp']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


def objective(trial):
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid", "elu", "selu"])
    n_layers = trial.suggest_int("n_layers", 2, 4)  # 👈 Entre 2 et 4 couches

    model = Sequential()
    
    # Layer 1 (toujours présente)
    model.add(Dense(trial.suggest_int("units1", 32, 256), activation=activation, input_shape=(X_train.shape[1],)))
    model.add(Dropout(trial.suggest_float("dropout1", 0.1, 0.5)))

    # Layer 2 (toujours présente)
    model.add(Dense(trial.suggest_int("units2", 32, 256), activation=activation))
    model.add(Dropout(trial.suggest_float("dropout2", 0.1, 0.5)))

    # Layer 3 (optionnelle)
    if n_layers >= 3:
        model.add(Dense(trial.suggest_int("units3", 32, 256), activation=activation))
        model.add(Dropout(trial.suggest_float("dropout3", 0.1, 0.5)))

    # Layer 4 (optionnelle)
    if n_layers == 4:
        model.add(Dense(trial.suggest_int("units4", 32, 256), activation=activation))
        model.add(Dropout(trial.suggest_float("dropout4", 0.1, 0.5)))

    # Output layer
    model.add(Dense(1))

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        callbacks=[early_stop],
        verbose=1
    )

    return min(history.history["val_loss"])
# Run the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=70)

# Best hyperparameters
print("Best trial:")
print(study.best_trial.params)

# Save best hyperparameters to JSON
with open("best_params.json", "w") as f:
    json.dump(study.best_trial.params, f)
