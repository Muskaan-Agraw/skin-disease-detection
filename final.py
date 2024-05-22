import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


st.title('SkinSense: Face Dermatology Classifier')
st.text('''This is a application which classifies the disease on your face into three classes which are :
    1. Acne
    2. Redness
    3. Bags''')

st.markdown("# What are these conditions?")

acne_text = """Acne is a common skin condition that causes pimples, and it usually appears on the face, 
shoulders, arms, legs, and buttocks. Acne occurs when tiny holes on the skin's surface, called pores, become clogged. 
The blockage is called a plug or comedone, and it can be caused by a mixture of oil and skin cells."""

bags_text = """Bags on the face can be caused by a number of factors, including aging, lifestyle, and UV exposure. 
As people age, the skin around their eyes loses elasticity, causing it to sag and form bags. Other lifestyle factors that can 
contribute to the formation of bags include: Poor sleep, Dehydration, Excessive salt intake, Alcohol consumption, Smoking, 
and UV exposure."""

redness_text = """Rosacea is a long-term inflammatory skin condition that causes reddened skin and a rash, 
usually on the nose and cheeks. It may also cause eye problems. The symptoms typically come and go, with many people 
reporting that certain factors, such as spending time in the sun or experiencing emotional stress, bring them on."""

def text_img_collection(text, img):
    st.write(text)
    st.write("")
    st.markdown("##### Pictures from dataset")
    st.image(img,width=700)
    

disease_options = ["Acne", "Redness", "Bags"]
selected_disease_option = st.selectbox("Select disease to know about", disease_options)

if selected_disease_option == "Acne":
    text_img_collection(acne_text, "pictures/output_acne.jpg")

if selected_disease_option == "Bags":
    text_img_collection(bags_text, "pictures/output_bags.jpg")

if selected_disease_option == "Redness":
    text_img_collection(redness_text, "pictures/output_redness.jpg")


st.markdown("# Know your skin")
# picture_options = ["Click a Picture", "Upload Picture"]
# selected_picture_option = st.selectbox("Select option for picture",picture_options)



model = tf.keras.models.load_model('skin_disease_model.hdf5')


def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((150, 150)) 
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction



# if selected_picture_option=="Click a Picture":
#     img_file_buffer = st.camera_input("Take a picture")
#     if img_file_buffer:
#         st.image(img_file_buffer, width=700)

# if selected_picture_option == "Upload Picture":
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        # Make prediction
        predictions = predict(image)
        disease_array = ["Atopic Dermatitis", "Bullous Disease", "Lupus and other Connective Tissue Disease", "Melanoma Skin Cancer Nevi and Moles", "Scabies Lyme Disease and other Infestations and Bites", "Vascular Tumor"]
        st.markdown("### Predition Probablity of different classes :")
        prediction = predictions[0]
        # st.write(prediction.shape)
        st.write(f"{disease_array[0]} : {prediction[0]}")
        st.write(f"{disease_array[1]} : {prediction[1]}")
        st.write(f"{disease_array[2]} : {prediction[2]}")
        st.write(f"{disease_array[3]} : {prediction[3]}")
        st.write(f"{disease_array[4]} : {prediction[4]}")
        st.write(f"{disease_array[5]} : {prediction[5]}")
        st.write("")
        max_index = np.argmax(prediction)
        st.markdown(f"### This Disese belongs to _{disease_array[max_index]}_ class.")


