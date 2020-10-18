FROM tensorflow/tensorflow:latest-gpu
RUN mkdir image_search
WORKDIR image_search
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]