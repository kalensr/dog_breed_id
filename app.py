import os
import base64
from io import BytesIO
from fastai.vision import *
from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import BadRequest


def evaluate_image(img) -> str:
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class


app = Flask(__name__)
# path = '/home/kalensr/ml/web/dog_breed_app/model/'
path = 'model/'
# learn = load_learner(path).to_fp16()
learn = load_learner(path)


# img = open_image(path + "weeping/00000002.jpg")

@app.route('/', methods=['GET'])
def index():
    """Render the app"""
    return render_template('serving_template.html')


@app.route('/image', methods=['POST'])
def eval_image() -> str:
    """Evaluate the image!"""
    input_file = request.files.get('file')
    print(type(input_file))
    if not input_file:
        return BadRequest("File is not present in the request")
    if input_file.filename == '':
        return BadRequest("File is not present in the request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png', 'heif')):
        return BadRequest("Invalid file type: " + input_file.filename)

    input_buffer = BytesIO()

    # The following is a workaround for werkzeug bug
    # https://github.com/pallets/werkzeug/issues/1733
    # previously the following statement used.  input_file.save(input_buffer)
    from shutil import copyfileobj
    copyfileobj(input_file.stream, input_buffer, 16384)

    predict = str(evaluate_image(open_image(input_buffer)))
    return jsonify({
        'predict': str(predict[predict.find('-') + 1:])
    })


if __name__ == "__main__":
    app.run(debug="On", host='0.0.0.0', threaded=False, port="8010")
