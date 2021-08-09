from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask import Flask, request, jsonify, render_template
from queue import Queue, Empty
from threading import Thread
import time

app = Flask(__name__)
model_path = "EasthShin/BTS_Lyrics_GPT-Neo-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)
classifier = pipeline('text-generation', model_path, tokenizer=tokenizer)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

print("complete model loading")

def handle_requests_by_batch():
    while True:
        request_batch = []
        while not(len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))

            except Empty:
                continue
            for requests in request_batch:
                try:
                    requests["output"] = make_answer(requests["inputs"])
                except Exception as e:
                    requests["output"] = e

handler = Thread(target=handle_requests_by_batch).start()

def make_answer(context):
    try:
        result = dict()
        result[0] = classifier(context)[0]

        return result

    except Exception as e:
        print('Error occur in generating text!', e)
        return jsonify({'error': e}), 500


@app.route('/generating', methods=['POST'])
def generate():
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        args = []
        context = request.form['context']
        args.append(context)
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    req = {'inputs': args}
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)
    return req['output']

@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return "Clear", 200

@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200

@app.route('/')
def main():
    return render_template('index.html'), 200

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')