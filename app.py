from flask import Flask, render_template , request ,abort, flash, url_for, redirect
import secrets
import os
import numpy as np 
import cv2
from yolo_predect import *

#Load yolo 
nets,Lables,Colors = startModel()

app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret

@app.route("/")
def home(): 
    return render_template('home.html')

#Upload file and result image detected on
@app.route("/upload/")
def upload():
	return render_template('upload.html')

@app.route("/result/",methods = ['GET','POST'])
def result():
	if request.method == 'POST':
		file = request.files['file']

		# Save the file to ./static/images
		file_name = file.filename
		#check type of file
		f_finish = str.split(file_name,'.')[1]
		if f_finish not in ('jpg','jpge','png'):
			flash("Uploaded file is not a vailid image!\nPlease try again!")
			return redirect(url_for("upload"))
		file.save(os.path.join("static/images",file_name))

		# get image to predect
		image = np.array(cv2.imread(os.path.join("static/images",file_name)))
		img , time_predect, result = predection(image,nets,Lables,Colors)
		# Save detected image
		cv2.imwrite(os.path.join("static/detected",file_name),img)
	try:
		return render_template("result.html",display_image = file_name, display_time = time_predect,display_result = result)
	except FileNotFoundError:
		abort(404)




if __name__ == '__main__':
    app.run(debug=True)