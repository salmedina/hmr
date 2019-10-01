import argparse
import flask
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import os
import json
from PIL import ImageDraw, Image

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

def draw_bboxex(img_path, bboxes, selected_indices, save_path):
    im = Image.open(img_path)

    for idx, bbox in enumerate(bboxes):
        if idx in selected_indices:
            draw = ImageDraw.Draw(im)
            print(bbox)
            draw.rectangle(bbox, fill=None)

    # write to stdout
    im.save(save_path)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def main():
    display_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meva_sample_3.png')
    return render_template('index.html', display_image=display_image_path)

@app.route('/renderbbox', methods=['GET'])
def render_bbox():
    display_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meva_sample_3.png')
    save_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meva_render.png')
    print(save_image_path)

    persons_selected = request.args.getlist('jsdata[]')
    persons_selected = [int(person) for person in persons_selected]
    print(persons_selected)
    person_bboxes = [[319, 43, 539, 427], [0, 2, 245, 485], [95, 132, 429, 551]]

    draw_bboxex(display_image_path, person_bboxes, persons_selected, save_image_path)

    return json.dumps({"url": save_image_path}), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='Server IP')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    args = parser.parse_args()

    app.run(debug=True, host=args.ip, port=args.port)