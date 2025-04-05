from flask import Flask, request, jsonify, render_template
import chatbot
import detection
import pyttsx3

MODEL_PATH = r"C:\Users\mhmdf\Documents\freelance\recipe_chatbot\models\best_908.pt"
CONFIDENCE = 0.45

last_prompt=""

app = Flask(__name__)

model = detection.load_model(MODEL_PATH)

def weight_popup():
    warn = False
    msg = "less weight detected!"
    output = detection.check_shelf(model)
    for i in output['weights']:
        if i<50:
            warn = True
            msg = msg + f", weight:{i}g on gauge {output['weights'].index(i)}"
    return jsonify({'show_alert': warn, 'message': msg})

def check_user_input(messege):
    if "shelf" in messege.lower():
        print("checking the shelf...")
        msg = messege + f" Available items response from Ai shelf: {detection.check_shelf(model)}"    
    else:
        msg = messege
    return msg


def speak(text, speed=130):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    default_voice_id = voices[1].id
    engine.setProperty('rate', speed)
    engine.setProperty('voice', default_voice_id)
    
    try:
        engine.say(f"{text}\r\n")
        engine.runAndWait()
        if not engine._inLoop:
            engine.endLoop()
        engine = None
    except Exception as e:
        print(f"Error: {e}")

def get_bot_response(user_message):
    global last_prompt
    msg=check_user_input(user_message)
    print(msg)
    last_prompt = chatbot.chat_with_gpt(msg)
    return last_prompt

@app.route('/')
def home():
    return render_template('chat_ui.html')

@app.route('/get_response', methods=['POST'])
def respond():
    user_message = request.json.get('message', '')
    
    response = get_bot_response(user_message)
    
    return jsonify({'response': response})

@app.route('/voice_feedback', methods=['POST'])
def voice_feedback():
    global last_prompt
    data = request.json
    speak(last_prompt)
    print(f"Requested: {data}")
    return '', 204 

if __name__ == '__main__':
    app.run(debug=True)