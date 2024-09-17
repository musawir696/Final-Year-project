from flask import Flask, request, jsonify, render_template
from detection_model import detect_hallucination  # Import the detection function

app = Flask(__name__)  # Make sure this is defined before using @app.route

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    question = data['question']
    answer = data['answer']
    
    hallucinated, hallucinated_part = detect_hallucination(question, answer)
    
    return jsonify({
        'hallucinated': hallucinated,
        'hallucinated_part': hallucinated_part
    })

if __name__ == '__main__':
    app.run()

