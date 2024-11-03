from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# Creating a Flask Instance
app = Flask(__name__)

IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'public/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")
model = load_model('models/pneu_cnn_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_preprocessor(path):
    print('Processing Image ...')
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, IMAGE_SIZE)
    normalized_img = resized_img / 255.0
    reshaped_img = np.reshape(normalized_img, (1, 150, 150, 3))
    return reshaped_img

def model_predict(image):
    print("Image shape:", image.shape)
    print("Image dimension:", image.ndim)
    prediction = model.predict(image)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['imagefile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            print(f"Image saved at {img_path}")
            image = image_preprocessor(img_path)
            pred = model_predict(image)
            result = 'Positive' if pred >= 0.5 else 'Negative'
            confidence = f'{(pred * 100 if pred >= 0.5 else (1 - pred) * 100):.2f}%'
            classification = f'{result} ({confidence})'
            return render_template('index.html', prediction=classification, imagePath=img_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
