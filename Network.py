import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import time
import datetime
from tensorboard.plugins.hparams import api as hp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

n_of_layers = None


def mnist_cnn_model():
    global n_of_layers
    image_size = 28
    num_channels = 1  # 1 for grayscale images
    num_classes = 10  # Number of outputs
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     padding='same', input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers

    print('''\nПолный список функций-активаций: 
deserialize(...) - Returns activation function given a string identifier.
elu(...): Exponential Linear Unit.
exponential(...): Exponential activation function.
gelu(...): Applies the Gaussian error linear unit (GELU) activation function.
get(...): Returns function.
hard_sigmoid(...): Hard sigmoid activation function.
linear(...): Linear activation function (pass-through).
relu(...): Applies the rectified linear unit activation function.
selu(...): Scaled Exponential Linear Unit (SELU).
serialize(...): Returns the string identifier of an activation function.
sigmoid(...): Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
softmax(...): Softmax converts a real vector to a vector of categorical probabilities.
softplus(...): Softplus activation function, softplus(x) = log(exp(x) + 1).
softsign(...): Softsign activation function, softsign(x) = x / (abs(x) + 1).
swish(...): Swish activation function, swish(x) = x * sigmoid(x).
tanh(...): Hyperbolic tangent activation function.\n''')
    func_activation_dense = input('Введите название функции активации скрытого слоя. Рекомендуем relu --> ')
    n_of_layers = int(input('Введите количество слоёв нейросети. Рекомендован 1 --> '))
    for i in range(n_of_layers):
        num_of_neurons = int(input('Введите количество нейронов на скрытом слое. Рекомендуем 128 --> '))
        model.add(Dense(num_of_neurons, activation=func_activation_dense))
    # Output layer
    func_activation_out = input('Введите название функции активации выходного слоя. Рекомендуем softmax --> ')
    model.add(Dense(num_classes, activation=func_activation_out))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def mnist_cnn_train(model):
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

    # Get image size
    image_size = 28
    num_channels = 1  # 1 for grayscale images

    # re-shape and re-scale the images data
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    # re-shape and re-scale the images validation data
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    num_of_epochs = int(input('Введите количество эпох обучения. Рекомендуем 2 --> '))
    print("Обучаем и тренируем сеть...")
    t_start = time.time()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, {
        'num_relu_units': 512,
        'dropout': 0.2})

    # Start training the network
    model.fit(train_data, train_labels_cat, epochs=num_of_epochs, batch_size=64,
              validation_data=(val_data, val_labels_cat), callbacks=[tensorboard_callback, hparams_callback])
    time_of_learning = round(time.time() - t_start, 2)
    print("Готово!", f"Нейросеть обучилась за {int(time_of_learning // 60)} мин {round(time_of_learning % 60)} секунд")

    return model


model = mnist_cnn_model()
print(model.summary())
mnist_cnn_train(model)
model.save('cnn_digits_28x28.h5')

# weights = model.get_layer(f'dense_1').get_weights()[0]
# print(weights[0])
# weights[0][0] = -0.04
# print(weights[0])


def cnn_digits_predict(model, image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file,
                                             target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))
    result = model.predict_classes([img_arr])
    return result[0]


model = tf.keras.models.load_model('cnn_digits_28x28.h5')
ANSWER = cnn_digits_predict(model, 'resized.jpg')
