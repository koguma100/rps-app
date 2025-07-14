import os
import base64
import random
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = YOLO('model/best.pt')  # Ensure this is your trained YOLOv8 model
choices = ['rock', 'paper', 'scissor']

def get_winner(user, computer):
    user = user.lower()
    computer = computer.lower()
    if user == computer:
        return 'Draw'
    elif (user == 'rock' and computer == 'scissor') or \
         (user == 'scissor' and computer == 'paper') or \
         (user == 'paper' and computer == 'rock'):
        return 'You Win!'
    else:
        return 'You Lose!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.form.get('image')
    if not data_url:
        return jsonify({'error': 'No image data received'})

    # Decode base64 image
    header, encoded = data_url.split(',', 1)
    image_data = base64.b64decode(encoded)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with open(filepath, 'wb') as f:
        f.write(image_data)

    # Run YOLO prediction
    results = model(filepath)
    boxes = results[0].boxes

    if boxes and len(boxes.cls) > 0:
        class_id = int(boxes.cls[0].item())
        prediction = results[0].names[class_id]
    else:
        prediction = 'unknown'

    # Random computer move
    computer_choice = random.choice(choices)
    result = get_winner(prediction, computer_choice)

    return jsonify({
        'user_move': prediction,
        'computer_move': computer_choice,
        'result': result,
        'image_url': filepath
    })

if __name__ == "__main__":
    app.run()
