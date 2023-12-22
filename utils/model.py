from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import io

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def handle_question_answering(image_data, question):
  image = Image.open(io.BytesIO(image_data))
  
  # prepare inputs
  encoding = processor(image, question, return_tensors="pt")
  
  # forward pass
  outputs = model(**encoding)
  logits = outputs.logits
  idx = logits.argmax(-1).item()
  return model.config.id2label[idx]

if __name__ == '__main__':
  print('utils')