import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

#Loading DATASET
ipl = pd.read_csv('ipl_data.csv')
ipl.head()

df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'],axis=1)

X = df.drop(['total'],axis=1)
Y = df['total']

venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()
 
# Fit and transform the categorical features with label encoding
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().transform(X_test)

model = keras.Sequential([keras.layers.Input( shape=(X_train_scaled.shape[1],)),keras.layers.Dense(512, activation='relu'),keras.layers.Dense(216, activation='relu'),keras.layers.Dense(1, activation='linear')])
hubber_loss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam',loss=huber_loss)

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))

