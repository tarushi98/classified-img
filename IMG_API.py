import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from flask import Flask,render_template, request,jsonify
import requests
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut', 'gray_leaf_spot', 'leaf_rust', 'northern_leaf_blight', 'stem_rust']

def getlabel(img):
    path=r"model3.h5"
    model = keras.models.load_model(path)
    img_array = tf.expand_dims(img, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    x=100*np.max(score)
    if(x<60):
        ans=({"ans":"Your plant is healthy."})
    else:
        ans = ({"ans":"This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))})
    return ans

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def processreq():
    content = request.json
    image_url = content["image_url"]
    img_data = requests.get(image_url).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)
    img_data = cv2.imread("image_name.jpg")
    img_data = cv2.resize(img_data,(299,299))
    label = getlabel(img_data)
    return jsonify(label)



if __name__=='__main__':
    app.run(debug=True)
