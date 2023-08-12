from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (4, 4), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (4, 4) , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3)))

# Adding a third convolutional layer
classifier.add(Conv2D(16, (4, 4) , activation = 'relu'))



# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dense(units = 32, activation = 'relu'))

classifier.add(Dense(units = 16, activation = 'relu'))

classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Datasets\Train',
                                                 target_size = (64, 64),
                                                 batch_size = 128,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets\Test',
                                            target_size = (64, 64),
                                            batch_size = 128,
                                            class_mode = 'categorical')

model = classifier.fit(training_set,
                         steps_per_epoch = 4000,
                         epochs = 2,
                         validation_data = test_set,    
                         validation_steps = 2000)

classifier.save("model.h5")
print("Saved model to disk")


# Part 3 - Making new predictions




import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load and preprocess the image for prediction
test_image = image.load_img('cat1.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Normalize the image

# Make a prediction
result = model.predict(test_image)

# Get the class indices from the training set
class_indices = training_set.class_indices

# Get the predicted class label
predicted_class_index = np.argmax(result)
predicted_class = list(class_indices.keys())[predicted_class_index]

print("Predicted class:", predicted_class)



