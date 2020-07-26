import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import csv
import pandas as pd

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy")>0.9988:
            print("Reached 99%")
            self.model.stop_training = True


callback = Callback()
(train_image,train_labels),(test_image,test_labels) = mnist.load_data()

print(train_image[1].shape)
train_image = tf.expand_dims(train_image,axis=-1)
test_image = tf.expand_dims(test_image,axis=-1)
train_image = train_image/255
test_image = test_image/255

print(test_image[1].shape)
print(train_labels)
labels=[]
image = []
with open("train.csv",'r') as f:
    file_1 = csv.reader(f)
    next(file_1)
    for row in file_1:
        labels.append(row[0])
        image_1 = row[1:]
        image.append(np.array_split(image_1,28))
    f.close()
image_2 = []
with open("test.csv",'r') as f_1:
    file_2 = csv.reader(f_1)
    next(file_2)
    for row in file_2:
        image_1 = row[0:]
        image_2.append(np.array_split(image_1,28))

image_2 = np.array(image_2).astype('float32')
image_2 = image_2/255
image_2 = tf.expand_dims(image_2,axis=-1)
images = np.array(image).astype('float32')
images = images/255
images = tf.expand_dims(images,axis=-1)
print(images.shape)
print(train_image.shape)
labels = np.array(labels)
print(labels.shape)
print(train_labels.shape)
train_data_total = tf.concat([train_image,images],axis=0)
total_labels_total = tf.concat([train_labels,labels],axis=0)
print(train_data_total.shape)
print(total_labels_total)
total_labels_total = tf.keras.utils.to_categorical(total_labels_total)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy']
)
test_labels = tf.keras.utils.to_categorical(test_labels)

model.fit(train_data_total,total_labels_total,validation_data=(test_image,test_labels),epochs=20,callbacks=[callback])

predictions = model.predict(image_2)
prediction_test = []

for i in predictions:
    prediction_test.append(np.argmax(i))

my_submission=pd.DataFrame({'ImageId':[],'Label':[]},dtype=int)

for x in range(len(predictions)):
    my_submission=my_submission.append({'ImageId':x+1,'Label':prediction_test[x]},ignore_index=True)

my_submission.to_csv("cnn_mnist_datagen.csv",index=False)