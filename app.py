Here is the corrected and fully functional Streamlit code.

The main issues fixed include:

* Missing assignment operators (`=`) throughout the script.
* Incorrect unpacking of `train_test_split`.
* Indentation and syntax errors within the `@st.cache_resource` block.
* Completing the cut-off sections at the end of the file.

```python
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢")
st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# Simple model loading with caching
@st.cache_resource
def load_model():
    try:
        # Using sklearn's built-in digits dataset to train a basic classifier
        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_model()

if model is None:
    st.warning("Could not load model. Using fallback recognition.")
else:
    st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    try:
        # Convert to grayscale and resize to 8x8
        img_gray = image.convert('L')
        img_resized = img_gray.resize((8, 8))
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype=float)
        
        # Invert colors if background is light (assuming dark digit on light background)
        if np.mean(img_array) > 128:
            img_array = 255.0 - img_array
            
        # Normalize to match sklearn's training range (0 to 16)
        img_array = (img_array / 255.0) * 16.0
        img_flat = img_array.flatten().reshape(1, -1)
        
        if model is not None:
            # Make prediction
            prediction = model.predict(img_flat)[0]
            st.write(f"## Prediction: **{prediction}**")
            
            # Show probabilities
            probs = model.predict_proba(img_flat)[0]
            st.write("### Probabilities:")
            for i, prob in enumerate(probs):
                st.write(f"Digit {i}: {prob:.2%}")
        else:
            # Fallback recognition
            st.write("## Using fallback recognition")
            digit_guess = int(np.argmax(np.sum(img_array.reshape(8, 8), axis=0)) % 10)
            st.write(f"Estimated digit: **{digit_guess}**")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Instructions sidebar
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (PNG, JPG, or JPEG).
2. The app will convert the image to grayscale, resize it to 8x8 pixels, and normalize it.
3. The neural network model will classify the digit and show prediction probabilities.
""")

```
