import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Plant Disease Detection", page_icon=":leaves:")
st.set_option('deprecation.showfileUploaderEncoding', False)

# Cache the model loading to improve performance
@st.cache(allow_output_mutation=True)
def load_plant_disease_model():
    # Update the path to the model if necessary
    model_path = 'best_model.h5'
    model = load_model(model_path)
    return model

model = load_plant_disease_model()

st.title("PLANT DISEASE DETECTION SYSTEM :leaves:")
st.markdown("_____")
st.write("Plant disease, an impairment of the normal state of a plant that interrupts or modifies its vital functions. All species of plants, wild and cultivated alike, are subject to disease. Although each species is susceptible to characteristic diseases, these are, in each case, relatively few in number. The occurrence and prevalence of plant diseases vary from season to season, depending on the presence of the pathogen, environmental conditions, and the crops and varieties grown. Some plant varieties are particularly subject to outbreaks of diseases while others are more resistant to them.")
st.write("Common methods for the diagnosis and detection of plant diseases include visual plant disease estimation by human raters, microscopic evaluation of morphology features to identify pathogens, as well as molecular, serological, and microbiological diagnostic techniques.")
st.write("To learn more about Plant Disease, you can [click here](https://cropwatch.unl.edu/soybean-management/plant-disease#:~:text=A%20plant%20disease%20is%20defined,abiotic%20and%20biotic%20plant%20diseases.)")
st.markdown("____")

file = st.file_uploader("UPLOAD IMAGE :arrow_heading_down:", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.array(image) / 255.0  # Normalize the image
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Thank you for your time")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    class_index = np.argmax(predictions)
    string = f"This image is most likely to be of: {class_names[class_index]}"
    st.success(string)

st.markdown("____")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Libraries used while building this website are:")
    st.write(
    """
    - Streamlit
    - PIL
    - Matplotlib
    - TensorFlow
    - Numpy
    - Pandas
    """
    )
