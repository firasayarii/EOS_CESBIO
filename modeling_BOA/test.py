import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda


tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

data=pd.read_csv('no_outliers.csv')
data['BOA_fraction']=data['BOA_RT']/1000
scaler=StandardScaler()
scaler_y=StandardScaler()
cols_to_scale = data.drop(columns=['BOA_fraction']).columns
X_scaled = scaler.fit_transform(data[cols_to_scale])
y_scaled=scaler_y.fit_transform(data['BOA_fraction'].values.reshape(-1, 1))
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.25,random_state=42)



# Chargement des paramètres optimaux
with open("best_params.json", "r") as f:
    params = json.load(f)

# Fonction d'activation personnalisée qui ne reçoit les 6 paramètres qu'en sortie
def transmission_activation(inputs):
    alpha, m_g, m_a, dtau_gas_scat, dtau_aero_scat, T_gas_abs, T_aero_abs = inputs  # Décomposer la liste d'inputs
    
    # Dénominateur
    denom = 1 + alpha * dtau_gas_scat * m_g + (alpha / 3.0) * dtau_aero_scat * m_a
    
    # Absorption
    T_abs = tf.pow(T_gas_abs, m_g) * tf.pow(T_aero_abs, m_a)
    
    # Transmission finale
    return T_abs / denom

# Définition des entrées complètes
input_features = Input(shape=(X_train.shape[1],))  # Inclut toutes les colonnes pour l'entraînement

# Couches cachées
hidden = Dense(params["units1"], activation=params["activation"])(input_features)
hidden = Dropout(params["dropout1"])(hidden)

hidden = Dense(params["units2"], activation=params["activation"])(hidden)
hidden = Dropout(params["dropout2"])(hidden)

if params["n_layers"] >= 3:
    hidden = Dense(params["units3"], activation=params["activation"])(hidden)
    hidden = Dropout(params["dropout3"])(hidden)

if params["n_layers"] == 4:
    hidden = Dense(params["units4"], activation=params["activation"])(hidden)
    hidden = Dropout(params["dropout4"])(hidden)

# Transmission directe de la sortie avant activation
alpha = Lambda(lambda x: x)(hidden)  # Passage direct sans transformation

# Entrées des 6 paramètres après entraînement
input_m_g = Input(shape=(1,))
input_m_a = Input(shape=(1,))
input_dtau_gas_scat = Input(shape=(1,))
input_dtau_aero_scat = Input(shape=(1,))
input_T_gas_abs = Input(shape=(1,))
input_T_aero_abs = Input(shape=(1,))

# Application de la fonction analytique sur la sortie après entraînement
# Application de la fonction analytique sur la sortie
output = Lambda(lambda inputs: transmission_activation(inputs), output_shape=(None, 1))(
    [alpha, input_m_g, input_m_a, input_dtau_gas_scat, input_dtau_aero_scat, input_T_gas_abs, input_T_aero_abs]
)

# Définition du modèle final
model = Model(inputs=[input_features, input_m_g, input_m_a, input_dtau_gas_scat, input_dtau_aero_scat, input_T_gas_abs, input_T_aero_abs], outputs=output)

# Compilation du modèle
optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

# Affichage du résumé du modèle
model.summary()

X_train_m_g = X_train["mprime_g"].values.reshape(-1, 1)
X_train_m_a = X_train["mprime_a"].values.reshape(-1, 1)
X_train_dtau_gas_scat = X_train["GOD"].values.reshape(-1, 1)
X_train_dtau_aero_scat = X_train["AODS"].values.reshape(-1, 1)
X_train_T_gas_abs = X_train["Tg_abs"].values.reshape(-1, 1)
X_train_T_aero_abs = X_train["Ta_abs"].values.reshape(-1, 1)

history = model.fit(
    [X_train, X_train_m_g, X_train_m_a, X_train_dtau_gas_scat, X_train_dtau_aero_scat, X_train_T_gas_abs, X_train_T_aero_abs], 
    y_train,
    batch_size=32,
    epochs=100
)