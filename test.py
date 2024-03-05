# from tensorflow.keras.preprocessing import image
# These lines import the required libraries for working with images and NumPy arrays.
from tensorflow.keras.models import load_model
import numpy as np


# Loading the pre-trained model:
# This line loads the pre-trained model from the 'model.h5' file. The model is assumed to have been trained and saved previously.
model = load_model('model.h5')
from tensorflow.keras.preprocessing import image
# testing the model
# Defining a function testing_image to classify an input image:
# This function takes the file path of an image as input.
# Loading the image using Keras' image.load_img and resizing it to the required input size (224x224):
def testing_image(image_directory):
# It loads the image and resizes it to the dimensions expected by the model.
    test_image = image.load_img(image_directory, target_size = (224, 224))
    # Converting the image to a NumPy array and expanding the dimensions to match the model's input shape:
    # This converts the image into a NumPy array and adds an extra dimension to match the expected input shape of the model.
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    # Normalizing pixel values to be in the range [0, 1]:
    test_image = test_image/255
    # Making predictions using the loaded model:
    result = model.predict(x= test_image)
    print(result)
    # Determining the class with the highest probability and assigning a corresponding label:
    # This block of code checks the index of the highest probability in the prediction results and assigns a corresponding class label.
    if np.argmax(result)  == 0:
      prediction = 'Covid 19'
    elif np.argmax(result)  == 1:
      prediction = 'Normal'
    elif np.argmax(result)  == 2:
      prediction ='Pneumonia'
    
    else:
      prediction = 'Unknown'
      # It prints the final predicted class label.
    print(prediction)
    
# Calling the function with a specific image path:
# This line calls the testing_image function with the path to a specific image file
# to test the model's prediction. The result will be printed in the console.

# testing_image(r'D:\Covid\archive\Data\test\COVID19\COVID19(466).jpg')
# testing_image(r'D:\Covid\archive\Data\train\NORMAL\NORMAL(8).jpg')
testing_image(r'C:\Users\user\Desktop\covid\archive\Data\test\PNEUMONIA\PNEUMONIA(3426).jpg')