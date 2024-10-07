from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
tf_model = Sequential()
tf_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(32, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(32, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Flatten())
tf_model.add(Dense(100, activation='relu'))
tf_model.add(Dense(1, activation='sigmoid'))
tf_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Preprocess and setup the data
train = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test = ImageDataGenerator(rescale=1./255)
train_path = r"D:\ML_Projects\Face-Mask-Detection-System\Data\train"
train_img = train.flow_from_directory(train_path, target_size=(150,150), batch_size=16, class_mode='binary')
test_path = r"D:\ML_Projects\Face-Mask-Detection-System\Data\test"
test_img = test.flow_from_directory(test_path, target_size=(150,150), batch_size=16, class_mode='binary')

# Train and test the model
mask_model = tf_model.fit(train_img, epochs=10, validation_data=test_img)

# Save the model
tf_model.save(r"D:\ML_Projects\Face-Mask-Detection-System\Models\mask_model_1.h5", mask_model)
