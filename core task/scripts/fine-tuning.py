import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_PATH = "../data/huge data/"

image_size = (800, 400)
batch_size = 2

category_type = "rock"
training_folder = "UV"

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH + '/prepared data/' + category_type +
    '/training data/' + training_folder + '/train/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_PATH + '/prepared data/' + category_type +
    '/training data/' + training_folder + '/validation/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = keras.models.load_model(
    "../models/rock/non-UV-InceptionResNetV2-02-0.41.h5")

model.layers[0].trainable = True

fine_tune_at = 200

for layer in model.layers[0].layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

filepath = "../models/" + category_type + "/" + training_folder + \
    "-InceptionResNetV2-Fine-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_fine = [checkpoint]

epochs = 50
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history_fine = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    workers=4,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list_fine)
