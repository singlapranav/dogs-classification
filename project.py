import streamlit as st
from ultralytics import YOLO
from PIL import Image


st.set_page_config(layout="wide")
    

tab1, tab2 = st.tabs(["intro", "classification"])

with tab1:

    st.title("Dog Classification :dog:")
    
    st.write("\n")
    st.write("\n")
    st.write("\n")

    st.subheader('My Introduction', divider='rainbow')
    st.write("My name is Pranav Singla. I am studying in class 11th CBSE from Heritage Xperiential School Gurgaon")

    st.subheader('App introduction', divider='rainbow')
    st.write("This is a dog breed prediction app that I have created with the help of ultralytics YOLO model training")



with tab2:
    st.title("Dog Classification :dog:")

    st.write("\n")
    st.write("\n")
    st.write("\n")

    @st.cache_resource
    def my_model():
        model = YOLO("best.pt")
        return model
    
    st.subheader('Upload your dog image to find its breed!', divider='rainbow')
    img = st.file_uploader("")

    if img is not None:
        im = Image.open(img)
        st.image(im)
        mod = my_model()
        res = mod.predict(im)
        tmp = res[0].probs.top5
        conf = res[0].probs.top5conf
        conf = conf.tolist()
        col = st.columns(2)
        with col[0]:
            for i in tmp:
                st.write(res[0].names[i])
        with col[1]:
            for i in conf:
                i = i * 100
                st.write(str(i))
                st.write('%')

        st.write("\n")
        st.write("\n")
        st.write("\n")
        prediction = st.radio(
        "Was the prediction correct?",
        ["Yes :blush:", "No :pensive:"],)
        

    

    

                
                
