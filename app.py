import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Configure paths
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
class_labels = {
    0: 'Glioma',
    1: 'Healthy',
    2: 'Meningioma',
    3: 'Pituitary'
}

# Load models
simple_cnn_model = tf.keras.models.load_model('/simple_cnn_brain_tumor_model.h5')
mobilenetv2_model = tf.keras.models.load_model('/mobilenetv2_brain_tumor_model.h5')
resnet50_model = tf.keras.models.load_model('/resnet50_brain_tumor_model.h5')
vgg16_model = tf.keras.models.load_model('/vgg16_brain_tumor_model.h5')
alexnet_model = tf.keras.models.load_model('/alexnet_brain_tumor_model.h5')
autoencoder_model = tf.keras.models.load_model('/autoencoder_brain_tumor_model.h5')
transformer_model = tf.keras.models.load_model('/cnn_transformer_brain_tumor_model.h5')


# Heuristic check for MRI images
def is_mri_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode not in ['L', 'RGB']:
            return False
        img_np = np.array(img)
        if img.mode == 'RGB':
            mean_diff = np.mean(np.abs(img_np[..., 0] - img_np[..., 1])) + \
                        np.mean(np.abs(img_np[..., 1] - img_np[..., 2]))
            if mean_diff > 20:
                return False
        return True
    except:
        return False

# Prediction helper
def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    prediction_distribution = {
        class_labels[i]: f"{pred * 100:.2f}%" for i, pred in enumerate(predictions[0])
    }

    return predicted_label, confidence, prediction_distribution

# Load model accuracy from text files
def load_accuracy(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip() + "%"
    except:
        return "N/A"

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

        # MRI check
        if not is_mri_image(file_path):
            return render_template('index.html', error="❌ The uploaded image does not appear to be a brain MRI scan. Please upload a valid MRI image.")

        try:
            simple_cnn_pred, simple_cnn_conf, simple_cnn_dist = predict_image(simple_cnn_model, file_path)
            mobilenetv2_pred, mobilenetv2_conf, mobilenetv2_dist = predict_image(mobilenetv2_model, file_path)
            resnet50_pred, resnet50_conf, resnet50_dist = predict_image(resnet50_model, file_path)
            vgg16_pred, vgg16_conf, vgg16_dist = predict_image(vgg16_model, file_path)
            alexnet_pred, alexnet_conf, alexnet_dist = predict_image(alexnet_model, file_path)
            autoencoder_pred, autoencoder_conf, autoencoder_dist = predict_image(autoencoder_model, file_path)
            transformer_pred, transformer_conf, transformer_dist = predict_image(transformer_model, file_path)
        except Exception as e:
            return f"Prediction failed: {str(e)}"

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
            'resnet50': {
                'predicted_label': resnet50_pred,
                'confidence': f"{resnet50_conf * 100:.2f}%",
                'prediction_distribution': resnet50_dist
            },
            'vgg16': {
                'predicted_label': vgg16_pred,
                'confidence': f"{vgg16_conf * 100:.2f}%",
                'prediction_distribution': vgg16_dist
            },
            'alexnet': {
                'predicted_label': alexnet_pred,
                'confidence': f"{alexnet_conf * 100:.2f}%",
                'prediction_distribution': alexnet_dist
            },
            'autoencoder': {
                'predicted_label': autoencoder_pred,
                'confidence': f"{autoencoder_conf * 100:.2f}%",
                'prediction_distribution': autoencoder_dist
            },
            'transformer': {
                'predicted_label': transformer_pred,
                'confidence': f"{transformer_conf * 100:.2f}%",
                'prediction_distribution': transformer_dist
            }
        }

        accuracies = {
            'simple_cnn': load_accuracy('/accuracy/cnn_accuracy.txt'),
            'mobilenetv2': load_accuracy('/accuracy/mobilenetv2_accuracy.txt'),
            'resnet50': load_accuracy('/accuracy/resnet50_accuracy.txt'),
            'vgg16': load_accuracy('/accuracy/vgg16_accuracy.txt'),
            'alexnet': load_accuracy('/accuracy/alexnet_accuracy.txt'),
            'autoencoder': load_accuracy('/accuracy/autoencoder_accuracy.txt'),
            'transformer': load_accuracy('/accuracy/cnn_transformer_accuracy.txt')
        }

        return render_template('index.html', results=results, accuracies=accuracies, file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
