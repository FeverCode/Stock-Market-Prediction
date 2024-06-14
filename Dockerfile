FROM python:3.9.19-bookworm
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
EXPOSE 5000
