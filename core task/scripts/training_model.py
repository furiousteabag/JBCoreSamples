"""Training on a sample data.

Methods to prepare generators, build and train model.

  Typical usage example:

    train_generator, validation_generator, test_generator = prepare_generators(path="../data/huge data/",
                                                                               image_size=(
                                                                                   224, 224),
                                                                               batch_size=8,
                                                                               category_type="rock",
                                                                               training_folder="non-UV")
    model = build_model(image_size=train_generator.target_size,
                        num_classes=train_generator.num_classes)

    history = train_model(model=model,
                          train_generator=train_generator,
                          validation_generator=validation_generator,
                          epochs=50,
                          model_name="VGG16",
                          category_type="rock",
                          training_folder="non-UV")
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_generators(path: str,
                       image_size: (int, int),
                       batch_size: int,
                       category_type: str,
                       training_folder: str):
    """
    Args:
        path: Path to huge data folder.
        training_folder: UV or non-UV folder.
    """

    train_datagen = ImageDataGenerator(rescale=1/255,)
    validation_datagen = ImageDataGenerator(rescale=1/255,)
    test_generator = ImageDataGenerator(rescale=1/255,)

    train_generator = train_datagen.flow_from_directory(
        path + '/prepared data/' + category_type +
        '/training data/' + training_folder + '/train/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        path + '/prepared data/' + category_type +
        '/training data/' + training_folder + '/validation/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_generator.flow_from_directory(
        path + '/prepared data/' + category_type +
        '/training data/' + training_folder + '/test/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def build_model(image_size: (int, int),
                num_classes: int,
                optimizer="adam",
                loss="categorical_crossentropy"):
    """Initializes model structure and complies it.
    """

    base_model = tf.keras.applications.vgg16.VGG16(
        input_shape=(image_size[0],
                     image_size[1],
                     3),
        include_top=False,
        weights='imagenet')

    print("Number of layers in base model: {}".format(len(base_model.layers)))

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    return model


def train_model(model,
                train_generator,
                validation_generator,
                epochs: int,
                model_name: str,
                category_type: str,
                training_folder: str):
    """Trains model and saves intermediate results.
    Args:
        model_name: Name of pre-trained model.
        category_type: Category, used for choosing folder
            for checkpoints
        training_folder: UV or non-UV. Used for naming
            trained model.
    """

    batch_size = train_generator.batch_size
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    filepath = "../models/" + category_type + "/" + training_folder + \
        "-" + model_name + "-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        workers=4,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list)

    return history


def main():
    train_generator, validation_generator, test_generator = prepare_generators(path="../data/huge data/",
                                                                               image_size=(
                                                                                   224, 224),
                                                                               batch_size=8,
                                                                               category_type="rock",
                                                                               training_folder="non-UV")
    model = build_model(image_size=train_generator.target_size,
                        num_classes=train_generator.num_classes)

    history = train_model(model=model,
                          train_generator=train_generator,
                          validation_generator=validation_generator,
                          epochs=50,
                          model_name="VGG16",
                          category_type="rock",
                          training_folder="non-UV")

    return history


if __name__ == "__main__":
    main()
