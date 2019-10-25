"""Fine-tuning on a sample data.

Methods to fine-tune model.

  Typical usage example:

    train_generator, validation_generator, _ = prepare_generators(path="../data/huge data/",
                                                                  image_size=(
                                                                      224, 224),
                                                                  batch_size=8,
                                                                  category_type="rock",
                                                                  training_folder="non-UV")
    model = build_model(path="../models/rock/non-UV-VGG16-01-0.66.h5",
                        fine_tune_at=4)

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

    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)

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

    test_generator = test_datagen.flow_from_directory(
        path + '/prepared data/' + category_type +
        '/training data/' + training_folder + '/test/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def build_model(path: str,
                fine_tune_at: int,
                optimizer="adam",
                loss="categorical_crossentropy"):
    """Loads model, unfreezes its layers and compiles it.
    Args:
        fine_tune_at: Number of last unfreezed layers.
    """

    model = keras.models.load_model(path)

    model.layers[0].trainable = True

    print("Number of layers in base model: {}".format(len(model.layers[0].layers)))

    for layer in model.layers[0].layers[:-fine_tune_at]:
        layer.trainable = False

    for layer in model.layers[0].layers:
        print("{}: {}".format(layer, layer.trainable))

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
        "-" + model_name + "-FineTuned" + "-{epoch:02d}-{val_acc:.2f}.h5"
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
    train_generator, validation_generator, _ = prepare_generators(path="../data/huge data/",
                                                                  image_size=(
                                                                      224, 224),
                                                                  batch_size=8,
                                                                  category_type="rock",
                                                                  training_folder="non-UV")
    model = build_model(path="../models/rock/non-UV-VGG16-23-0.70.h5",
                        fine_tune_at=2,
                        optimizer=keras.optimizers.Adam(lr=0.000001))

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
