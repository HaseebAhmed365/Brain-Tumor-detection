from keras.models import load_model
import cv2
import json
import gradio as gr


model_result=load_model("fyp.h5",compile=True)

f=open("fyp file.json")
data=json.load(f)

Tumor_Classes=list(data)


def Tumor_Prediction(image):
  image=cv2.resize(image,(32,32))/255.0
  result=model_result.predict(image.reshape(1,32,32,3)).argmax()

  return Tumor_Classes[result],data[Tumor_Classes[result]]['Description'],data[Tumor_Classes[result]]['Causes'],data[Tumor_Classes[result]]['Symptoms'],data[Tumor_Classes[result]]['Treatment']


interface=gr.Interface(fn=Tumor_Prediction,
                       inputs="image",
                       outputs=[gr.components.Textbox(label="Tumor Name"),gr.components.Textbox(label="Description"),gr.components.Textbox(label="Causes"),gr.components.Textbox(label="Symptoms"),gr.components.Textbox(label="Treatment")],
                       enable_queu=True)
interface.launch(debug=True)
    
