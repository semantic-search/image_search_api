FROM tensorflow/tensorflow:latest-gpu
RUN mkdir image_search
WORKDIR image_search
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn main:app --reload --host 0.0.0.0 --port 7000