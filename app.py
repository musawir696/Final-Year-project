from flask import Flask, request, jsonify, render_template, redirect, url_for
import os

app = Flask(__name__)  # Make sure this is defined before using @app.route

USER_FILE = 'users.txt'

# Ensure the file exists
if not os.path.exists(USER_FILE):
    open(USER_FILE, 'w').close()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    from detection_model import detect_hallucination  # Import inside to avoid potential circular imports
    data = request.get_json()
    question = data['question']
    answer = data['answer']
    
    hallucinated, hallucinated_part = detect_hallucination(question, answer)
    
    return jsonify({
        'hallucinated': hallucinated,
        'hallucinated_part': hallucinated_part
    })

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    with open(USER_FILE, 'r') as f:
        users = [line.strip().split(':') for line in f.readlines()]
    
    for user, passw in users:
        if username == user and password == passw:
            return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid username or password'})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    # Check if user already exists
    with open(USER_FILE, 'r') as f:
        users = [line.strip().split(':')[0] for line in f.readlines()]
    if username in users:
        return jsonify({'success': False, 'error': 'User already exists'})
    
    # Save new user
    with open(USER_FILE, 'a') as f:
        f.write(f'{username}:{password}\n')
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
