import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import io
import base64 # Import base64 for image encoding

# Model path - TensorFlow SavedModel directory (exported from Keras 3)
MODEL_PATH = "waste_model_export"

# Mapping waste types
biodegradable = ['biological', 'paper', 'cardboard', 'clothes']
non_biodegradable = ['plastic', 'metal', 'green-glass', 'brown-glass', 'white-glass', 'battery', 'trash', 'shoes']

# Waste information dictionary
waste_info = {
    "plastic": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Check the plastic type and recycle it if possible."},
    "metal": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Take it to a metal scrap or recycling center."},
    "cardboard": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Recycle or reuse it for packing or crafts."},
    "paper": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Keep it dry and put it in the paper recycling bin."},
    "clothes": {"Reusable": "Yes", "Recyclable": "No", "Disposal": "Donate if in good condition, or use a textile bin."},
    "biological": {"Reusable": "No", "Recyclable": "No", "Disposal": "Compost it or put it in the biodegradable waste bin."},
    "battery": {"Reusable": "No", "Recyclable": "Yes", "Disposal": "Drop it at an e-waste collection point safely."},
    "trash": {"Reusable": "No", "Recyclable": "No", "Disposal": "Throw it in the general waste bin."},
    "shoes": {"Reusable": "Yes", "Recyclable": "No", "Disposal": "Donate if wearable or put in a textile bin."},
    "green-glass": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Rinse and put it in the glass recycling bin."},
    "brown-glass": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Clean and recycle it if not broken."},
    "white-glass": {"Reusable": "Yes", "Recyclable": "Yes", "Disposal": "Recycle or reuse it if it's in good shape."}
}

# CRITICAL: Class Names Order - Verified from train_generator.class_indices
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# Create bio_map for binary classification
bio_map = {}
for name in CLASS_NAMES:
    if name in biodegradable:
        bio_map[name] = 'biodegradable'
    elif name in non_biodegradable:
        bio_map[name] = 'non_biodegradable'

# Flask App Setup
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model globally - Using TensorFlow SavedModel (Keras 3 export)
model = None
model_fn = None
input_key = None

try:
    # Load the SavedModel
    loaded = tf.saved_model.load(MODEL_PATH)
    model_fn = loaded.signatures['serving_default']
    
    # Get the input key
    input_keys = list(model_fn.structured_input_signature[1].keys())
    input_key = input_keys[0]
    
    # Get output keys
    output_keys = list(model_fn.structured_outputs.keys())
    
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    print(f"Input key: {input_key}")
    print(f"Output keys: {output_keys}")
    
    model = True  # Flag to indicate model is loaded
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Helper Functions
def allowed_file(filename):
    """Checks if a file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """Loads, resizes, and prepares the image for the model."""
    img = Image.open(image_file)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Convert to tensor
    img_tensor = tf.constant(img_array, dtype=tf.float32)
    
    return img_tensor

# NEW HELPER FUNCTION
def get_base64_data_url(file_stream, mime_type):
    """Converts a file stream to a Base64 Data URL string."""
    file_stream.seek(0) # Go back to the start of the stream
    encoded_string = base64.b64encode(file_stream.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"


def model_predict(image_tensor):
    """Runs prediction and formats the output."""
    if model is None:
        return {'error': 'Model not loaded. Please check server logs.'}
    
    try:
        # Run inference using the correct input key
        predictions = model_fn(**{input_key: image_tensor})
        
        # Get the output (should be 'output_0')
        pred = predictions['output_0'].numpy()
        
        # Get the predicted class
        class_idx = np.argmax(pred[0])
        confidence = np.max(pred[0])
        waste_type = CLASS_NAMES[class_idx]
        
        # Debug output
        print(f"\n=== Prediction ===")
        print(f"Predicted: {waste_type} (index {class_idx})")
        print(f"Confidence: {confidence:.2%}")
        print(f"Top 3 predictions:")
        top_3_indices = np.argsort(pred[0])[-3:][::-1]
        for idx in top_3_indices:
            print(f"  {CLASS_NAMES[idx]}: {pred[0][idx]:.2%}")
        
        # Get biodegradability classification
        bio_label = bio_map.get(waste_type, 'non_biodegradable')
        
        # Get disposal suggestions
        suggestion = waste_info.get(waste_type.lower(), {
            'Reusable': 'Unknown',
            'Recyclable': 'Unknown',
            'Disposal': 'No disposal info available.'
        })
        
        return {
            'waste_type': waste_type.replace('-', ' ').title(),
            'bio_non_bio': bio_label.replace('_', '-').title(),
            'confidence': f"{confidence * 100:.2f}%",
            'reusable': suggestion['Reusable'],
            'recyclable': suggestion['Recyclable'],
            'disposal': suggestion['Disposal']
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Prediction failed: {str(e)}'}

# Routes
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    initial_error = "⚠ Model not loaded. Check server logs." if model is None else None
    return render_template('index.html', error=initial_error)

@app.route('/predict', methods=['POST'])
def upload():
    """Handles image upload and classification."""
    if model is None:
        return render_template('index.html', error='Model not loaded. Check server logs.')
    
    # Check if file is present
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected.')
    
    if file and allowed_file(file.filename):
        try:
            # 1. Read file into a stream and save a copy of the stream
            img_stream = io.BytesIO(file.read())
            
            # 2. Get the MIME type for the data URL
            # Note: A simple way, may need refinement for robust MIME detection
            ext = file.filename.rsplit('.', 1)[1].lower()
            mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
            
            # 3. Generate the Base64 Data URL
            image_data_url = get_base64_data_url(img_stream, mime_type)
            
            # 4. Preprocess image for the model (uses the same stream)
            img_stream.seek(0) # IMPORTANT: Rewind the stream before preprocessing
            img_tensor = preprocess_image(img_stream)
            
            # 5. Get prediction
            prediction_result = model_predict(img_tensor)
            
            if 'error' in prediction_result:
                return render_template('index.html', error=prediction_result['error'])
            
            # 6. Pass both the prediction result and the image data URL
            return render_template(
                'index.html', 
                result=prediction_result, 
                image_data_url=image_data_url # <--- NEW VARIABLE
            )
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', error=f'Prediction failed: {str(e)}')
    else:
        return render_template('index.html', error='Invalid file type. Only JPG, JPEG, PNG are allowed.')

# Run App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)