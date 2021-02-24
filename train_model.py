import numpy as np
import preprocess as pp
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES']='-1'

def split_data(X,y,ratio=0.2):
	split=int((1-ratio)*len(X))
	X_train, y_train = X[:split], y[:split]
	X_test, y_test = X[split:], y[split:]
	return X_train,y_train,X_test,y_test
	
	
X,y = pp.read_data()
X,y = pp.preprocess_data(X,y)
X_train,y_train,X_test,y_test = split_data(X,y)

X_train = X_train.reshape(X_train.shape[0],96,96,1)
X_test = X_test.reshape(X_test.shape[0],96,96,1)

model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(filters=8,kernel_size=(5,5),strides=3,activation='relu',input_shape=(96,96,1)),
		tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
		tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=3,activation='relu'),
		tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics='accuracy')

model.fit(X_train,y_train,batch_size=100,epochs=200,validation_split=0.1)
model.evaluate(X_test,y_test)

model.save('models/model1')
