import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import json
import datetime
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
import h5py
print(h5py.__version__)

print(tf.__version__)
'''df=pd.read_csv('preprocessed_data.csv')
X=df.drop(columns=['alpha_exp'])
y=np.log(df['alpha_exp'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# Load best parameters from JSON file
with open("best_params.json", "r") as f:
    params = json.load(f)

# Build the model
model = Sequential()

# Layer 1
model.add(Dense(params["units1"], activation=params["activation"], input_shape=(28,)))
model.add(Dropout(params["dropout1"]))

# Layer 2
model.add(Dense(params["units2"], activation=params["activation"]))
model.add(Dropout(params["dropout2"]))

# Layer 3 (if applicable)
if params["n_layers"] >= 3:
    model.add(Dense(params["units3"], activation=params["activation"]))
    model.add(Dropout(params["dropout3"]))

# Layer 4 (if applicable)
if params["n_layers"] == 4:
    model.add(Dense(params["units4"], activation=params["activation"]))
    model.add(Dropout(params["dropout4"]))

# Output layer
model.add(Dense(1))
# Compile model
optimizer = Adam(learning_rate=params["lr"])
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 1. Early stopping
early_stop = EarlyStopping(
    monitor='val_mae',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# 2. Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 3. Model checkpoint (save best model)
checkpoint_cb = ModelCheckpoint(
    filepath='deep_model.h5',
    monitor='val_mae',
    save_best_only=True,
    verbose=1
)

# 4. CSV logger (log training history)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
csv_logger = CSVLogger(f'training_log_{timestamp}.csv')

# Train phase :

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=params["batch_size"],
    callbacks=[early_stop, reduce_lr, checkpoint_cb, csv_logger],
    verbose=1
)'''














'''bins_a = np.load("mprime_a_bins.npy")
bins_g = np.load('mprime_g_bins.npy')
encoder=joblib.load('encoder_OH.pkl')
scaler=joblib.load('scaler_SS.pkl')
df_no_outliers=pd.read_csv('dart_use.csv')
cols_to_scale = df_no_outliers.drop(columns=['alpha','mprime_a_bin','mprime_g_bin']).columns
cols_to_encode = ['mprime_a_bin', 'mprime_g_bin']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_no_outliers[cols_to_scale])

# Encodage one-hot
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_bins_encoded = encoder.fit_transform(df_no_outliers[cols_to_encode])

# Récupération des noms des colonnes encodées
encoded_feature_names = encoder.get_feature_names_out(cols_to_encode)

# Construction du DataFrame final
df_scaled = pd.DataFrame(
    np.concatenate([X_scaled, X_bins_encoded], axis=1),
    columns=list(cols_to_scale) + list(encoded_feature_names),
    index=df_no_outliers.index  # Pour conserver les bons index
)

joblib.dump(scaler, 'scaler_SS.pkl')

# Save the OneHotEncoder
joblib.dump(encoder, 'encoder_OH.pkl')
df_scaled.to_csv('dart_result.csv',index=False)'''