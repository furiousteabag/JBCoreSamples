import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_PATH = "../data/huge data/"

image_size = (400, 400)
batch_size = 8

category_type = "carbonate"
training_folder = "non-UV"

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH + '/prepared data/' + category_type +
    '/training data/' + training_folder + '/test/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_PATH + '/prepared data/' + category_type +
    '/training data/' + training_folder + '/test/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = tf.keras.applications.InceptionResNetV2(
    input_shape=(image_size[0],
                 image_size[1], 3),
    include_top=False,
    weights='imagenet')

print(len(base_model.layers))

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

epochs = 50
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

filepath = "../models/" + category_type + "/" + training_folder + \
    "-InceptionResNetV2-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    workers=4,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list)
