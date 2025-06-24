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
#df_no_outliers.drop(columns=['BOA_RT'],inplace=True)
#new = pd.read_csv('/home/fayari/Stage/modeling_BOA/add_training_v2.csv')
data=pd.concat([duplicated_data,df_no_outliers,df_no_outliers_SDB,df_no_outliers_2_SDB,augmented_data], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['BOA_fraction']=data['BOA_RT'] / 1000
data.drop(columns=['alpha','BOA_RT'],inplace=True)
# Séparer les features et la cible
'''y_add = data['alpha']
X_add = data.drop(columns=['alpha'])

# Paramètres
n_copies = 650
X_augmented = []
y_augmented = []

# Générer des copies bruitées
for _ in range(n_copies):
    noise = np.random.normal(loc=0.0, scale=0.01, size=X_add.shape)
    X_aug = X_add.values + noise
    X_augmented.append(X_aug)
    y_augmented.append(y_add.values)

# Empiler les données
X_augmented = np.vstack(X_augmented)
y_augmented = np.hstack(y_augmented)

# Reformer un DataFrame avec les colonnes originales
df_new_augmented = pd.DataFrame(
    np.hstack([X_augmented, y_augmented.reshape(-1, 1)]),
    columns=data.columns
)

# Fusionner avec df_no_outliers
df_augmented = pd.concat([df_no_outliers, df_new_augmented], ignore_index=True)
#bins_g = pd.cut(df_no_outliers['mprime_g'], bins=5, retbins=True)[1]
#bins_a = pd.cut(df_no_outliers['mprime_a'], bins=5, retbins=True)[1]
#df_no_outliers['mprime_g_bin'] = pd.cut(df_no_outliers['mprime_g'], bins=bins_g, labels=False, include_lowest=True)
#df_no_outliers['mprime_a_bin'] = pd.cut(df_no_outliers['mprime_a'], bins=bins_a, labels=False, include_lowest=True)
#np.save('mprime_g_bins_DART.npy', bins_g)
#np.save('mprime_a_bins_DART.npy', bins_a)

cols_to_scale = df_augmented.drop(columns=['alpha']).columns
#cols_to_encode = ['mprime_a_bin', 'mprime_g_bin']'''

# Standardisation
cols_to_scale = data.drop(columns=['BOA_fraction']).columns
scaler = StandardScaler()
scaler_y=StandardScaler()
X_scaled = scaler.fit_transform(data[cols_to_scale])
y_scaled=scaler_y.fit_transform(data['BOA_fraction'].values.reshape(-1, 1))
# Encodage one-hot
'''encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_bins_encoded = encoder.fit_transform(df_no_outliers[cols_to_encode])

# Récupération des noms des colonnes encodées
encoded_feature_names = encoder.get_feature_names_out(cols_to_encode)

# Construction du DataFrame final
df_scaled = pd.DataFrame(
    np.concatenate([X_scaled, X_bins_encoded], axis=1),
    columns=list(cols_to_scale) + list(encoded_feature_names),
    index=df_no_outliers.index  # Pour conserver les bons index
)'''
#joblib.dump(encoder, 'encoder_OH_DART.pkl')
joblib.dump(scaler, 'scaler_SS_BOA_DART.pkl')
joblib.dump(scaler_y, 'scaler_y_SS_BOA_DART.pkl')
'''df_scaled['alpha']=df_no_outliers['alpha']


X=df_scaled.drop(columns=['alpha'])
y=df_scaled['alpha']'''

#y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
#y_scaled = pd.Series(y_scaled.flatten(), index=y.index) 
#joblib.dump(scaler_y, 'scaler_y_DART.pkl')'''
X_train,X_val,y_train,y_val=train_test_split(X_scaled,y_scaled,test_size=0.2,random_state=1042)

### adding samples to training :

'''new=pd.read_csv('/home/fayari/Stage/modeling_BOA/add_training.csv')
new['mprime_g_bin'] = pd.cut(df_no_outliers['mprime_g'], bins=bins_g, labels=False, include_lowest=True)
new['mprime_a_bin'] = pd.cut(df_no_outliers['mprime_a'], bins=bins_a, labels=False, include_lowest=True)
X_scaled_add = scaler.transform(new[cols_to_scale])
X_bins_encoded_add = encoder.transform(new[cols_to_encode])
df_scaled_add = pd.DataFrame(
    np.concatenate([X_scaled_add, X_bins_encoded_add], axis=1),
    columns=list(cols_to_scale) + list(encoded_feature_names),
    index=new.index  # Pour conserver les bons index
)
y_add=new['alpha']
X_add=df_scaled_add
#X_train = pd.concat([X_train] + [df_scaled_add] * 70, ignore_index=True)

#y_add_scaled = scaler_y.transform(y_add.values.reshape(-1, 1))  # Passage à 2D pour scaler
#y_add_scaled = pd.Series(y_add_scaled.flatten(), index=y_add.index)  # Retour en Series
#y_train = pd.concat([y_train] + [y_add_scaled]*70, ignore_index=True)

n_copies = 700
X_augmented = []
y_augmented = []

for _ in range(n_copies):
    noise = np.random.normal(loc=0.0, scale=0.01, size=X_add.shape)  # petit bruit
    X_aug = X_add.values + noise
    X_augmented.append(X_aug)
    y_augmented.append(y_add.values)

# Concaténer
X_augmented = pd.DataFrame(np.vstack(X_augmented), columns=X_add.columns)
y_augmented = pd.Series(np.hstack(y_augmented))

X_train = pd.concat([X_train, X_augmented], ignore_index=True)
y_train = pd.concat([y_train, y_augmented], ignore_index=True)'''

# Load best parameters from JSON file
with open("best_params.json", "r") as f:
    params = json.load(f)

# Build the model
model = Sequential()

# Layer 1
model.add(Dense(params["units1"], activation=params["activation"], input_shape=(X_train.shape[1],)))
#model.add(Dropout(params["dropout1"]))

# Layer 2
model.add(Dense(params["units2"], activation=params["activation"]))

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
    filepath='deep_model_v2.h5',
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
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=params["batch_size"],
    callbacks=[early_stop, reduce_lr, checkpoint_cb, csv_logger],
    verbose=1
)

