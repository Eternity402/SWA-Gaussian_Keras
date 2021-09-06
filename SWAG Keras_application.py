# import libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm


# Download CIFAR100 dataset & preprocess


mnist = keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


# Define a simple model: similar to the keras tutorial's model. (Link - https://www.tensorflow.org/tutorials/images/cnn?hl=ko)


def Default_model():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(100, activation='softmax'))
    return model


# Compile settings


# Test for SGD
model = Default_model()
sgd = tf.keras.optimizers.SGD(learning_rate=0.003)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train SWA-Gaussian


theta_swag = None
theta_diag = None
theta_lowrank = []
num_theta = 0
K_rank = 10
cycle = 3
history = []
history_swag = []
for i in range(70):
    history.append(model.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs=1))
    theta = get_weights(model)
    if i % cycle == 0:
        if num_theta ==0:
            theta_swag = theta
            theta_diag = np.square(theta)
            num_theta+=1
            # ttheta_lowrank is a zero-vector.
        else:
            theta_swag = theta * (num_theta+1) + theta_swag * num_theta / (num_theta+1)
            theta_diag = np.square(theta) * (num_theta+1) + theta_diag * num_theta / (num_theta+1)
            if len(theta_lowrank) == K_rank:
                del theta_lowrank[0]
            theta_lowrank.append(theta - theta_swag)
        history_swag.append([theta_swag, theta_diag, theta_lowrank])


# SWA prediction


sgd_swa = []
for i in range(34):
    num_test = i
    bayes_pred = swa(history_swag[num_test][0],history_swag[num_test][1],history_swag[num_test][2], 20, test_images)
    prediction = np.argmax(bayes_pred,axis = 1)
    count = 0
    for i in range(10000):
        if test_labels[i]==prediction[i]:
            count +=1
    print(num_test, ':',count)
    sgd_swa.append(count/10000)


# SWAG prediction


sgd_swag = []
for i in range(34):
    num_test = i
    bayes_pred = bayesian_model_averaging(history_swag[num_test][0],history_swag[num_test][1],history_swag[num_test][2], 20, test_images)
    prediction = np.argmax(bayes_pred,axis = 1)
    count = 0
    for i in range(10000):
        if test_labels[i]==prediction[i]:
            count +=1
    print(num_test, ':',count)
    sgd_swag.append(count/10000)


# Plot


sgd_origin = []
for i in range(34):
    sgd_origin.append(history[i*3].history['val_accuracy'])
plt.plot(sgd_origin, label = 'sgd')
plt.plot(sgd_swag, label = 'swag')
plt.plot(sgd_swa, label = 'swa')
plt.legend()
plt.show()


# Function definition

def bayesian_model_averaging(theta_swag, theta_diag, theta_lowrank, num_sampling, input_data):
    model = Default_model()
    # above model is just used for check the output size
    for i in tqdm.tqdm(range(num_sampling)):
        theta_sample = draw_theta(theta_swag, theta_diag, theta_lowrank)
        if i==0:
            pred_total = pred_sample(theta_sample, input_data)/num_sampling
        else:
            pred_total +=pred_sample(theta_sample, input_data)/num_sampling
    return pred_total

def swa(theta_swag, theta_diag, theta_lowrank, num_sampling, input_data):
    model = Default_model()
    # above model is just used for check the output size
    for i in tqdm.tqdm(range(num_sampling)):
        theta_sample = theta_swag
        if i==0:
            pred_total = pred_sample(theta_sample, input_data)/num_sampling
        else:
            pred_total +=pred_sample(theta_sample, input_data)/num_sampling
    return pred_total

def draw_theta(theta_swag, theta_diag, theta_lowrank):
    z1 = np.random.normal(size = theta_diag.shape[0])
    z2 = np.random.normal(size = len(theta_lowrank))
    part1 = np.sqrt(theta_diag)
    theta_tilda = theta_swag + np.sqrt(theta_diag) * z1/np.sqrt(2) + np.matmul(np.array(theta_lowrank).T, z2)/np.sqrt(2 * (len(theta_lowrank)-1))
    return theta_tilda

def pred_sample(theta, input_data):
    model = rollback(theta)
    return model.predict(input_data)

def rollback(theta):
    index = 0
    model = Default_model()
    rollback_weights = []
    for i in range(len(model.weights)):
        num = np.array(model.weights[i]).flatten().shape[0]
        rollback_weights.append(theta[index:index+num].reshape(np.array(model.weights[i]).shape))
        index += num
    model.set_weights(rollback_weights)
    return model

def get_weights(model):
    theta_ = []
    for i in range(len(model.weights)):
        theta_.append(np.array(model.weights[i].value()).flatten())
    theta = np.hstack(theta_)
    return theta

