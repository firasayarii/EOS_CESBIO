import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.layers import Input, Dense, Dropout, Add
from tensorflow.keras.models import Model
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
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import joblib


tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
duplicated_data=pd.read_csv('/home/fayari/Stage/data_exploration_2/noisy_duplicated_data.csv')
df_no_outliers=pd.read_csv('/home/fayari/Stage/data_exploration/no_outliers.csv')
df_no_outliers_SDB = pd.read_csv('/home/fayari/Stage/data_exploration_2/no_outliers.csv')
df_no_outliers_2_SDB = pd.read_csv('/home/fayari/Stage/data_exploration_2/no_outliers_2.csv')
augmented_data=pd.read_csv('/home/fayari/Stage/data_augmentation/add_training.csv')
high_reflectance=pd.read_csv('/home/fayari/Stage/data_exploration_2/high_ref_data.csv')
test_cases=pd.read_csv('/home/fayari/Stage/Evaluation/test_cases.csv')
#df_no_outliers.drop(columns=['BOA_RT'],inplace=True)
#new = pd.read_csv('/home/fayari/Stage/modeling_BOA/add_training_v2.csv')
data=pd.concat([duplicated_data,df_no_outliers,df_no_outliers_SDB,df_no_outliers_2_SDB,augmented_data,high_reflectance,test_cases], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['BOA_fraction']=data['BOA_RT'] / 1000
data.drop(columns=['alpha','BOA_RT'],inplace=True)


# Standardisation
cols_to_scale = data.drop(columns=['BOA_fraction']).columns
scaler = StandardScaler()
scaler_y=StandardScaler()
X_scaled = scaler.fit_transform(data[cols_to_scale])
y_scaled=scaler_y.fit_transform(data['BOA_fraction'].values.reshape(-1, 1))

#joblib.dump(encoder, 'encoder_OH_DART.pkl')
joblib.dump(scaler, 'scaler_SS_BOA_DART.pkl')
joblib.dump(scaler_y, 'scaler_y_SS_BOA_DART.pkl')


X_train,X_val,y_train,y_val=train_test_split(X_scaled,y_scaled,test_size=0.2,random_state=1042)



# === Paramètres donnés ===
activation='relu'
n_layers = 3
units1 = 232
dropout1 = 0.11109031799752554
units2 = 249
dropout2 = 0.13781937549350962
units3 = 232
dropout3 = 0.13597114528177423
lr = 0.0003406646556719905
batch_size = 64

# === Construction ===
inputs = Input(shape=(X_train.shape[1],), name="Input_Layer")

# Layer 1
layer1 = Dense(800, activation=activation, name="Hidden_1")(inputs)
#layer1 = Dropout(dropout1, name="Dropout_1")(layer1)

# Layer 2
layer2 = Dense(400, activation=activation, name="Hidden_2")(layer1)
#x = Dropout(dropout2, name="Dropout_2")(x)

# Layer 3
x = Dense(800, activation=activation, name="Hidden_3")(layer2)
#x = Dropout(dropout3, name="Dropout_3")(x)

# Layer 4
x = Dense(400, activation=activation, name="Hidden_4")(x)
#x = Dropout(dropout3, name="Dropout_3")(x)
# === Skip connection : ajouter Layer 2 et Layer 4 ===
skip_added = Add(name="Skip_L2_L4")([x, layer2])

# Output
final_output = Dense(1,name="Output")(skip_added)

# Model
model = Model(inputs=inputs, outputs=final_output)

# === Compilation ===
optimizer = Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# === Callbacks ===
early_stop = EarlyStopping(
    monitor='val_mae',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

checkpoint_cb = ModelCheckpoint(
    filepath='deep_model_v2.h5',
    monitor='val_mae',
    save_best_only=True,
    verbose=1
)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
csv_logger = CSVLogger(f'training_log_{timestamp}.csv')

# === Entraînement ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr, checkpoint_cb, csv_logger],
    verbose=1
)

