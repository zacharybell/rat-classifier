from keras import layers, models

model = models.Sequential()

# model.add(layers.Conv2D())
# model.add(layers.MaxPooling2D())

# model.add(layers.Flatten())
# model.add(layers.Dense(0, activation='relu'))
# model.add(layers.Dense(4, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])