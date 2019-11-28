"""Training on a sample data.

Methods to prepare generators, build and train model.

  Typical usage example:

    run(base_model = keras.applications.vgg16.VGG16,
        image_size=(224, 224),
        batch_size=16,
        category_type="rock",
        training_folder="non-UV",
        optimizer=keras.optimizers.Adam(lr=0.00001),        
        model_name="VGG16")
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import wandb
from wandb.keras import WandbCallback



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

    train_datagen = ImageDataGenerator(rescale=1/255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")
    validation_datagen = ImageDataGenerator(rescale=1/255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")
    test_generator = ImageDataGenerator(rescale=1/255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")

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


def build_model(base_model_input,
                image_size: (int, int),
                num_classes: int,
                optimizer="adam",
                loss="categorical_crossentropy"):
    """Initializes model structure and complies it.
    """

    base_model = base_model_input(
        input_shape=(image_size[0],
                     image_size[1],
                     3),
        include_top=False,
        weights='imagenet')
    
    print("Number of layers in base model: {}".format(len(base_model.layers)))

    base_model.trainable = True

    set_trainable = False
    for layer in base_model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
            

    
    # base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
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

    time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    # Filepath for saving models.
    filepath = "../../models/" + category_type + "/" + training_folder + "/" + time + "/" \
         + model_name + "-epoch_{epoch:02d}-loss_{loss:.2f}-accuracy_{accuracy:.2f}-val_loss_{val_loss:.2f}-val_accuracy_{val_accuracy:.2f}.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath)

    # Filepath for TensorBoard logs.
    log_dir= "../../models/" + category_type + "/" + training_folder + "/" + time + "/" + "logs/"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # WanDB.
    wandb_dir= "../../models/" + category_type + "/" + training_folder + "/" + time + "/"
    wandb.init(project=category_type+'-'+training_folder, dir=wandb_dir)

    callbacks_list = [tb_callback, checkpoint, WandbCallback()]

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        workers=4,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list)

    return history


def run(model_name,
        image_size,
        batch_size,
        category_type,
        training_folder,
        base_model,
        path="../../data/huge data/",        
        optimizer=keras.optimizers.Adam(lr=0.00001),        
        loss="categorical_crossentropy",
        epochs=500):
    
    train_generator, validation_generator, test_generator = prepare_generators(path=path,
                                                                               image_size=image_size,
                                                                               batch_size=batch_size,
                                                                               category_type=category_type,
                                                                               training_folder=training_folder)
    model = build_model(base_model_input=base_model,
                        image_size=train_generator.target_size,
                        num_classes=train_generator.num_classes,
                        optimizer=optimizer)

    history = train_model(model=model,
                          train_generator=train_generator,
                          validation_generator=validation_generator,
                          epochs=epochs,
                          model_name=model_name,
                          category_type=category_type,
                          training_folder=training_folder)

    return history

def main():
    
    run(base_model = keras.applications.vgg16.VGG16,
        image_size=(224, 224),
        batch_size=16,
        category_type="rock",
        training_folder="non-UV",
        optimizer=keras.optimizers.Adam(lr=0.00001),        
        model_name="VGG16")


#if __name__ == "__main__":
#    main()
