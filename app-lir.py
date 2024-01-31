from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
import os
import ultralytics
from ultralytics import YOLO
import cv2
model = YOLO('best.pt')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
class UploadForm(FlaskForm):
    image = FileField('Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'gif'], 'Images only!')
    ])
@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if request.method == 'POST' and form.validate_on_submit():
        image = form.image.data
        image.save(os.path.join('static/uploads', image.filename))
        file = os.path.join('static/uploads', image.filename)
        results = model(source=file)
        global name
        img = results[0].plot()
        cv2.imwrite(os.path.join('static/returns', image.filename), img)
        if len(results[0].boxes.cls):
            name_dict = results[0].names
            name_list = []
            for a in results[0].boxes.cls:
                idx = int(a)
                name_list.append(name_dict[idx])
            name = ", ".join(name_list)
        else:
            name = 'Not Detected'
        return redirect(url_for('display_image', filename=image.filename))
    return render_template('upload_form.html', form=form)
@app.route('/display_image/<filename>')
def display_image(filename):
    return render_template('display_image.html', filename=filename, name=name)

@app.route('/file', methods=['GET', 'POST'])
def file():
    if request.method == 'POST':
        image = request.files['file']
        print(image.filename)
        image.save(image.filename)
        results = model(source=image.filename)
        global name
        if len(results[0].boxes.cls):
            img = results[0].plot()
            cv2.imwrite(os.path.join('/return', image.filename), img)
            name_dict = results[0].names
            name_list = []
            for a in results[0].boxes.cls:
                idx = int(a)
                name_list.append(name_dict[idx])
            name = ", ".join(name_list)
            return name
        else:
            name = 'Not Detected'
            return name

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9900, debug=True)