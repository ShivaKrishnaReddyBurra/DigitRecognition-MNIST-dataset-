import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('digit_classifier_model.keras')

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    if 'file' not in request.files:
        return jsonify({'error': 'Please upload an image file.'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    try:
        # Convert the file into a PIL image
       # Load the image
        img = Image.open(file).convert('L')
        img = np.array(img)
        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (28, 28))
        # Normalize the image
        img = img.astype('float32') / 255
        # Reshape the image to match the input shape of the model
        img = img.reshape((1, 28, 28, 1))
        # Load the trained model
        # Predict the digit
        prediction = model.predict(img)
        # Get the digit with the highest probability
        digit = np.argmax(prediction)


        # Return the prediction as JSON
        return jsonify(f'The predicted digit is: {digit}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
