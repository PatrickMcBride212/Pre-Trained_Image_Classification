from keras.applications import VGG16
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
import os
import time
import matplotlib.pyplot as plt
from keras.utils import plot_model
import keras
import pydot
import pydotplus
from pydotplus import graphviz
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def create_data_generators(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return train_datagen, test_datagen, train_generator, validation_generator


def plot_history(history, save_title, smooth=False):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    if smooth:
        plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
        plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
    else:
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    if smooth:
        plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
        plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
    else:
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    plt.savefig(save_title)


def create_new_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def generate_and_train_initial_model():
    model = create_new_model()
    if os.path.isdir('C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification'):
        train_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification/Reduced_Data/train'
        # test_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification/Reduced_Data/test'
        validation_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification/Reduced_Data/validation'
    else:
        train_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Reduced_Data/train'
        # test_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Reduced_Data/test'
        validation_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Reduced_Data/validation'

    start_time = time.time()

    train_datagen, test_datagen, train_generator, validation_generator = create_data_generators(train_dir, validation_dir)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    model.save('trained_model', include_optimizer=False)

    print("Runtime: %s" % (time.time() - start_time))

    plot_history(history, 'Generated_Graphs', True)


def fine_tune_model():
    if os.path.isdir('C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification'):
        train_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification/Reduced_Data/train'
        test_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification/Reduced_Data/test'
        validation_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Pre-Trained_Image_Classification/Reduced_Data/validation'
    else:
        train_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Reduced_Data/train'
        test_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Reduced_Data/test'
        validation_dir = 'C:/Users/mcbri/PycharmProjects/Pre-Trained_Image_Classification/Reduced_Data/validation'

    default_model_name = 'trained_model'
    while not os.path.exists(default_model_name):
        print("Warning, default model name not present in working directory.")
        print("Default model name: ", default_model_name)
        print("Please either generate a model, or specify the name of the model file you want to use.")
        valid_input = False
        while not valid_input:
            user_input = input("Type g for generation, or r for renaming. ")
            if user_input == 'g':
                print("Generating initial model:")
                generate_and_train_initial_model()
            elif user_input == 'r':
                default_model_name = input("Please enter the name of the model file you want to use: ")
                break
            else:
                print("Invalid input, try again")

    # now the path to the model file should exist, we can begin fine tuning.
    model = models.load_model(default_model_name)

    # Setting the last few layers of the vgg16 base to be trainable
    model.summary()
    model.get_layer(name='vgg16').summary()
    model.get_layer(name='vgg16').trainable = True
    set_trainable = False
    for layer in model.get_layer(name='vgg16').layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # creating the data augmentation again

    train_datagen, test_datagen, train_generator, validation_generator = create_data_generators(train_dir, validation_dir)

    # recompilation of the network with low learning rate

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=validation_generator,
                                  validation_steps=50)
    plot_history(history, 'Fine_Tuned_Graphs', True)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
    print('test acc:', test_acc)

    model.save('Fine-Tuned_Model', include_optimizer=False)


def get_model_architecture():
    user_input = False
    while not user_input:
        filename = input("Please enter filename to get architecture of: ")
        if os.path.isfile(filename):
            model = models.load_model(filename)
            model.summary()
            model.get_layer(name='vgg16').summary()
            plot_model(
                model,
                to_file="Model_Visualized.png",
                show_shapes=False,
                show_dtype=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=96)
            user_input = True
        else:
            print("File does not exist in working directory, please move here or input valid file name.")


def main():
    valid_input = False
    while not valid_input:
        print("Generate completely new model, fine-tune existing, or get model architecture?")
        user_input = input("Type g for generate, f for fine-tune, or a for get model architecture. ")
        if user_input == 'g':
            valid_input = True
            generate_and_train_initial_model()
        elif user_input == 'f':
            valid_input = True
            fine_tune_model()
        elif user_input == 'a':
            valid_input = True
            get_model_architecture()
        else:
            print("Warning, invalid input, try again.")


if __name__ == '__main__':
    main()
