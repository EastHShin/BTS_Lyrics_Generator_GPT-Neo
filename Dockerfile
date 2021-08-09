FROM python:3.8

RUN pip install flask transformers requests
RUN pip install torch --no-cache-dir

WORKDIR /app

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]