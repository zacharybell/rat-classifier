from keras.preprocessing.image import ImageDataGenerator

train_dir = ''
valid_dir = ''

train_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='categorical')

valid_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(valid_dir, target_size=(150, 150), batch_size=20, class_mode='categorical')