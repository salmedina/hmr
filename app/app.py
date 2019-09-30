import argparse
import flask
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import os

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def main():
    display_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meva_sample_3.png')
    return render_template('index.html', display_image=display_image_path)

@app.route('/renderbbox', methods=['GET'])
def render_bbox():
    persons_selected = request.args.getlist('jsdata[]')
    persons_selected = [int(person) for person in persons_selected]
    print(persons_selected)
    display_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meva_sample_3.png')
    return render_template('renderbbox.html', display_image=display_image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='Server IP')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    args = parser.parse_args()

    app.run(debug=True, host=args.ip, port=args.port)