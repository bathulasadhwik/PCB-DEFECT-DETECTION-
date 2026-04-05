FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

ENV PORT=8080
ENV FLASK_ENV=production

CMD ["python", "app.py"]