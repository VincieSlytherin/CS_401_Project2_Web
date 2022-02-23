#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
import pickle
import numpy as np
import os.path, time
import datetime



# In[ ]:

app = Flask(__name__)

with open('model_v1.pickle', 'rb') as f:
    
    app.version=pickle.load(f)
    app.time = pickle.load(f)#time
    app.model=pickle.load(f)#model
    

# app.version=pickle.load(open("model_v1.pickle","rb"))
# app.time=pickle.load(open("model_v1.pickle","rb"))
# app.model=pickle.load(open("model_v1.pickle","rb"))
app.vectorizer=pickle.load(open("vectorizer.pickle","rb"))
app.tf_transformer=pickle.load(open("tf_transformer.pickle","rb"))
# app.filePath = "model_v1.pickle"
# app.ModifiedTime=time.localtime(os.stat(app.filePath).st_mtime) #文件访问时间 



@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/api/american",methods=["POST"])
def success():
    
    app.content = request.form['text']

       
    text=np.array([app.content])

    text1=app.vectorizer.transform(text)
    text2=app.tf_transformer.transform(text1)
    predicted = app.model.predict(text2)
    app.prediction=predicted.tolist()
    
    return render_template("success.html",text=app.content,prediction=app.prediction,version=app.version,date=app.time)


#     return jsonify({"is_american":str(predicted[0]),"version":app.version,"model_date":app.time})