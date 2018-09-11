from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

train_dir = 'data/train'
valid_dir = 'data/validation'

train_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='categorical')

valid_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(valid_dir, target_size=(150, 150), batch_size=20, class_mode='categorical')


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=30,
    validation_data=valid_gen,
    validation_steps=50
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')

plt.figure()

plt.show()
