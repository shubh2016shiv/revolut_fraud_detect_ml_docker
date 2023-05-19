FROM python:3.9-slim

COPY . /fraud_backend_app

WORKDIR /fraud_backend_app

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT [ "python" ]
CMD ["backend_app.py"]