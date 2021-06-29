import os
from flask import Flask,render_template,request
from Display import Display
import cv2
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
@app.route('/')
def hello_world():
    return 'Hello World!'
@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        path = './models/06-28-15:10_RSIR/63.pth.tar'
        display = Display(path)
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f.filename), cv2.IMREAD_UNCHANGED)
        path_list = display.run(img, 'gf1mul')
        return render_template('result.html', path_list=path_list)

    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)