import base64
import cv2
import numpy as np
from skimage.feature import hog
from django.shortcuts import render
from app.forms import ImageUploadForm
import joblib

# Load the models
scaler = joblib.load('static/scaler.joblib')
one_class_svm_classifier = joblib.load('static/one_class_svm_classifier.joblib')
svm_classifier = joblib.load('static/svm_classifier.joblib')

# List of class names your model can predict
varieties = ['BR25', 'K-1', 'PBC_123']

def index(request):
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        uploaded_image = form.cleaned_data['image']
        with uploaded_image.open() as f:
            content = f.read()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file")

        # Convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to 128x128 pixels for uniformity
        resized_img = cv2.resize(gray_img, (128, 128))

        # Extract Histogram of Oriented Gradients (HOG) features
        image_features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=False)

        # Reshape for the scaler and the classifier
        image_features = np.array([image_features])

        # Feature scaling
        scaled_features = scaler.transform(image_features)

        # Perform prediction using One-Class SVM
        one_class_pred = one_class_svm_classifier.predict(scaled_features)

        if one_class_pred == -1:
            predicted_label = "UNRECOGNIZED IMAGE"
            accuracy = "N/A"
        else:
            # Perform prediction on the image
            probability_pred = svm_classifier.predict_proba(scaled_features)[0]
            predicted_label_index = np.argmax(probability_pred)
            predicted_label = varieties[predicted_label_index]
            accuracy = f'{probability_pred[predicted_label_index] * 100:.2f}%'

        # Convert the image to Base64 for displaying on the web
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        image_data = f'data:image/jpeg;base64,{encoded_image}'

        # Prepare the response
        context = {
            'image_data': image_data,
            'predicted_label': predicted_label,
            'accuracy': accuracy
        }
        return render(request, 'result.html', context)

    return render(request, 'index.html', {'form': form})
