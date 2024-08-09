import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

categories = ('Bmw', 'Mercedes', 'Volkswagen','Audi', 'Porsche')

def classify_image(img):
    pred, pre_idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=192, width =192)
label = gr.Label()
title = 'German Car Classifier'
description = "A Classifier that can distinguish between the 5 most common german manufactured cars, built using fastai"
examples = ['bmw.jpg','mercedes.jps','vw.jpg','audi.jpg','porsche.jpg']

demo = gr.Inteface(fn=classify_image, inputs=image, outputs=label, title=title, description=description, examples=examples)
demo.launch(share=True)