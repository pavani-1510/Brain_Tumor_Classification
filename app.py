import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure paths for saving uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels
class_labels = {
    0: 'Glioma',
    1: 'Healthy',
    2: 'Meningioma',
    3: 'Pituitary'
}

# Load pre-trained models
simple_cnn_model = tf.keras.models.load_model('/home/pavani-r/Documents/VSCODE/DL/Brain-Tumor-Classification/models/simple_cnn_brain_tumor_model.h5')
mobilenetv2_model = tf.keras.models.load_model('/home/pavani-r/Documents/VSCODE/DL/Brain-Tumor-Classification/models/mobilenetv2_brain_tumor_model.h5')
resnet50_model = tf.keras.models.load_model('/home/pavani-r/Documents/VSCODE/DL/Brain-Tumor-Classification/models/resnet50_brain_tumor_model.h5')
vgg16_model = tf.keras.models.load_model('/home/pavani-r/Documents/VSCODE/DL/Brain-Tumor-Classification/models/vgg16_brain_tumor_model.h5')

# Helper function for model prediction
def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    # print(predictions)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Calculate Prediction Distribution
    prediction_distribution = {class_labels[i]: f"{pred * 100:.2f}%" for i, pred in enumerate(predictions[0])}

    return predicted_label, confidence, prediction_distribution

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Evaluate for both models
        simple_cnn_pred, simple_cnn_conf, simple_cnn_dist = predict_image(simple_cnn_model, file_path)
        mobilenetv2_pred, mobilenetv2_conf, mobilenetv2_dist = predict_image(mobilenetv2_model, file_path)
        resnet50_pred, resnet50_conf, resnet50_dist = predict_image(resnet50_model, file_path)
        vgg16_pred, vgg16_conf, vgg16_dist = predict_image(vgg16_model, file_path)

        # Prepare results to pass to the template
        results = {
            'simple_cnn': {
                'predicted_label': simple_cnn_pred,
                'confidence': f"{simple_cnn_conf * 100:.2f}%",
                'prediction_distribution': simple_cnn_dist
            },
            'mobilenetv2': {
                'predicted_label': mobilenetv2_pred,
                'confidence': f"{mobilenetv2_conf * 100:.2f}%",
                'prediction_distribution': mobilenetv2_dist
            },
            'resnet50':{
                'predicted_label': resnet50_pred,
                'confidence': f"{resnet50_conf * 100:.2f}%",
                'prediction_distribution': resnet50_dist
            },
            'vgg16':{
            'predicted_label': vgg16_pred,
            'confidence': f"{vgg16_conf * 100:.2f}%",
            'prediction_distribution': vgg16_dist
            }
        }

        return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
