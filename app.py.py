# Importing necessary libraries
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# Loading the pre-trained model:

model = load_model('model.h5')
import numpy as np
import os
if not os.path.exists('uploads'):
    os.makedirs('uploads')
# Creating a Flask web application:
app = Flask(__name__)
# Defining a route for the home page:
# This route renders the 'index.html' template when the user visits the home page.
@app.route('/')
def index():
    return render_template('index.html')
# Defining a route for handling file uploads:
# This route is triggered when the user uploads an image via a POST request.
@app.route('/upload', methods=['POST'])


def upload():
    try:
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            print("Filename:", filename)

            # Save the file in the 'static' folder
            file.save(os.path.join('uploads', filename))
            path = "uploads/" + filename
            print("File saved at path:", path)

            test_image = image.load_img(path, target_size = (224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image/255
            result = model.predict(x= test_image)
            print(result)
            # Determining the predicted class label:
            # The code checks the index of the highest probability in the prediction results and assigns a corresponding class label.
            if np.argmax(result)  == 0:
              prediction ='COVID-19'
            elif np.argmax(result)  == 1:
              prediction = 'Normal'
            elif np.argmax(result)  == 2:
              prediction ='PNEUMONIA'

            print(prediction,"dddsaszdss")
        # Rendering the result in the HTML template:
        # The predicted class label is passed to the 'index.html' template.
            return render_template('index.html', Data=prediction)
        else:
            return render_template('index.html', Data="No file uploaded.")
    except Exception as e:
        # Print detailed exception information for debugging
        print("Error:", str(e))
        return render_template('index.html', Data="An error occurred while processing the file.")



# testing the model
# Testing the model on the uploaded image:
# This section preprocesses the uploaded image and uses the pre-trained model to make predictions.

# Running the Flask application:
# This ensures that the Flask application runs when the script is executed
if __name__ == '__main__':
    app.run()

# In summary, this Flask application allows users to upload an image,
# processes the image using a pre-trained model, and displays the prediction on the web page.
# The result is rendered using the 'index.html' template.