from flask import Flask, request, jsonify
from utils.model import handle_question_answering
from utils.ServerResponse import ServerResponse

app = Flask(__name__)
import json

# home route
@app.route('/')
def home():
    strHome = "Welcome VQA Home!"
    return strHome


@app.route('/predict_vqa', methods=['POST'])
def predict():
    try:
        # Get image
        if 'image' not in request.files or request.files['image'].filename == '':
            raise ValueError('Image invalid!')

        image_data = request.files['image'].read()

        # Get question
        data_question = request.form.get('data')

        if not data_question:
            raise ValueError('Question invalid!')

        question = json.loads(data_question).get('question')

        if question is None or question == '':
            raise ValueError('Question invalid!')

        # result
        result = handle_question_answering(image_data, question)

        response = ServerResponse('success', {'answer': result})
        return jsonify(response.__dict__), 200

    except ValueError as e:
        error_type = e.args[0].split()[0].lower()
        response = ServerResponse('error', {error_type: e.args[0]})
        return jsonify(response.__dict__), 400

if __name__ == '__main__':
    app.run(debug = True)