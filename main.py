from keras.applications import VGG16
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
import os
import time
import matplotlib.pyplot as plt


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


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

    plot_history(history)


def fine_tune_model():
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


def main():
    valid_input = False
    while not valid_input:
        print("Generate completely new model, or fine-tune existing?")
        user_input = input("Type g for generate, or f for fine-tune. ")
        if user_input == 'g':
            valid_input = True
            generate_and_train_initial_model()
        elif user_input == 'f':
            valid_input = True
            fine_tune_model()
        else:
            print("Warning, invalid input, try again.")


if __name__ == '__main__':
    main()
