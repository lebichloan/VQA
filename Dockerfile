FROM python:3.11

WORKDIR /vqa_server

COPY '/requirements.txt' .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]