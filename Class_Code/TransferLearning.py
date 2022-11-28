import tensorflow as tf

from tensorflow import keras

#from keras.applications.vgg16 import VGG16

#from keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import ResNet50

from keras.layers import Input

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import GlobalAveragePooling2D

from keras.models import Model




#from keras.layers import AveragePooling2D

#from keras.layers import Dropout




#From class

tf.compat.v1.disable_eager_execution()







res_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(32,32,3)))




# #'Freeze' the first two layers.

# for i in range(0, 2):

# res_model.layers[i].trainable = False




fine_tune = res_model.output

fine_tune = GlobalAveragePooling2D()(fine_tune)

#fine_tune = Dense(1000, activation='relu')(fine_tune)

fine_tune = Dense(1000, activation='relu')(fine_tune) #760 prev

fine_tune = Dense(43, activation='softmax')(fine_tune)




#Adam help from here: https://keras.io/api/optimizers/

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)




model = Model(inputs=res_model.input, outputs=fine_tune)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])




#This checkpoint code is from the keras documentation here: https://keras.io/api/callbacks/model_checkpoint/

checkpoint_path = '/tmp/checkpoint_res'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(

filepath=checkpoint_path,

save_weights_only=True,

monitor="val_accuracy",

mode="max",

save_best_only=True)




"""This created the model, which I now want to fit and test. I used the keras API web link to get information on how to do that: https://keras.io/api/models/model_training_apis/"""




#The next block actually peforms the model training.




#Code from here: https://keras.io/api/models/model_training_apis/




#Train the model on our training data:

#num_epochs = 30 #Maybe tTry more than 20 next?? #Maybe 36 or 37 next I think

num_epochs = 2 #Maybe tTry more than 20 next?? #Maybe 36 or 37 next I think




history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=num_epochs, validation_data=(x_valid, y_valid), callbacks=[model_checkpoint_callback])

final_loss, final_accuracy = model.evaluate(x_test, y_test)

print("The final loss was: ", final_loss)

print("Final accuracy was: ", final_accuracy)

#Also code from keras documentation on checkpoints

model.load_weights(checkpoint_path)

final_loss, final_accuracy = model.evaluate(x_test, y_test)

print("The final loss on best run was: ", final_loss)

print("Final accuracy on best run was: ", final_accuracy)




"""The block below plots the model data"""




#Plot the history of the model

#Help/Code from here: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

#Plot Accuracy

from matplotlib import pyplot as plt




plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("ResNet Model Accuracy over epochs")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train', 'test'], loc='upper left')

plt.show()




#Plot Loss

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("ResNet Model Loss over epochs")

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['train', 'test'], loc='upper left')

plt.show()




#And the final block saves that model out to my drive. (I was performing this computation on Google Colab).

#model.save('/content/drive/My Drive/adv_resnet_model')