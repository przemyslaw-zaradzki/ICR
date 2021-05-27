from posixpath import realpath
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField
from flask_uploads import configure_uploads, IMAGES, UploadSet
import os
from .image_processing import processing

app = Flask(__name__)

images_folder = os.path.join("static", "images")

app.config['SECRET_KEY'] = 'thisisasecrete'
app.config['UPLOADED_IMAGES_DEST'] = 'static/images'
app.config['DISPLAY_IMAGES_PATH'] = images_folder
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ["JPG"]
app.config['MAX_IMAGE_FILESIZE'] = 1 * 1024 * 1024

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    else:
        return False

def allowed_image_filesize(filesize):
    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False

images = UploadSet('images', IMAGES)
configure_uploads(app, images)

class MyForm(FlaskForm):
    image = FileField('image')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm()

    if form.validate_on_submit():
        filename = images.save(form.image.data)
        return f'Filename: {filename}'

    return render_template('index.html', form=form)


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            #if not allowed_image_filesize(request.cookies.get("filesize")):
            #    print("File exceeded maximum size")
            #    return redirect(request.url)
            image = request.files["image"]
            if image.filename == "":
                print("Image must have a filename")
                return redirect(request.url)
            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)
            else:
                filename = secure_filename(image.filename)
            #image.save(os.path.join(app.config["UPLOADED_IMAGES_DEST"], filename))
            image.save(app.config["UPLOADED_IMAGES_DEST"] + "\chessboard.jpg")
            processing(os.path.join(app.config["UPLOADED_IMAGES_DEST"], "chessboard.jpg"))
            return redirect(url_for('.result'))
    return render_template('upload_image.html')

@app.route("/result")
def result():
    pict = os.path.join(app.config["DISPLAY_IMAGES_PATH"], "chessboard.jpg")
    position = os.path.join(app.config["DISPLAY_IMAGES_PATH"], "current_board.png")
    return render_template('result.html', pict = pict, position = position)
