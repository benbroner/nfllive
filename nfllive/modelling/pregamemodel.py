import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('/Users/benbroner/nfllive/data/nflds1.csv')
def build_classifier(inputshape, outputshape): # makes very simply tensorflow regression neural network
	model = tf.keras.Sequential([
		keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[inputshape]),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(outputshape)
	])

	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
	return model

def train_model(df = data):

	y = df.h_win
	x = df.drop(columns=['h_win'])
	# x = StandardScaler().fit_transform(x)

	model = build_classifier(16, 1)
	model.fit(x, y, epochs=100)

# train_model(data)