import numpy as np
import mnist
from keras.models import Sequential
from keras. layers import Dense
from keras.utils import to_categorical

train_image = mnist.train_images()
train_labels = mnist.train_labels()
test_img = mnist.test_images()
test_labels  = mnist.test_labels()

#Normalizing

#[0,255] to [-0.5,0.5]

train_image = (train_image/255) - 0.5
test_img = (test_img/255) - 0.5

#Flatten 28X28

train_image = train_image.reshape((-1,784))
test_img = test_img.reshape((-1,784))

print(train_image.shape)
print(test_img.shape)

#fit model
#3 layers,2 layers with 64 ns 1 layer with 10 ns

model = Sequential()
model.add(Dense(64, activation='relu',input_dim=784))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_image,to_categorical(train_labels),epochs=3 ,batch_size=3)

model.evaluate(test_img,to_categorical(test_labels))

prdct = model.predict(test_img[:3])

print("The predicted numbers:")
print (np.argmax(prdct, axis =1))

print("The actual numbers:")
print(test_labels[:3])
