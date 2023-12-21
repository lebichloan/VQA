from flask import Flask, request, jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests

app = Flask(__name__)

# Load the Vilt model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.route('/')
def index():
   print('Wellcome to VQA')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        question = data.get('question')

        # Load the image from URL
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Prepare inputs
        encoding = processor(image, question, return_tensors="pt")

        # Perform inference
        outputs = model(**encoding)
        idx = outputs.logits.argmax(-1).item()
        predicted_answer = model.config.id2label[idx]

        return jsonify({'predictedAnswer': predicted_answer})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
