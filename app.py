import os
import json
import random
import uuid
from datetime import datetime
import requests
from dotenv import load_dotenv

from flask import Flask, request, jsonify, render_template, url_for

# Audio & STT libs
import soundfile as sf
try:
    import whisper
except ImportError:
    print("Whisper is not installed. Please run 'pip install openai-whisper'")
    whisper = None

# ------------------ Configuration ------------------
app = Flask(__name__)

load_dotenv()

API_KEY = os.getenv("IAPP_API_KEY")
TTS_URL = "https://api.iapp.co.th/thai-tts-kaitom2/tts"
AUDIO_FILENAME_PREFIX = 'tmp_user_audio_'
WHISPER_MODEL = 'small'
MODULES_FILE = 'modules.json'
OUTPUT_DIR = 'logs'
CACHE_DIR = os.path.join('static', 'audio')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

sessions = {}

# ------------------ Modules Loading ------------------
# CHANGED: Updated default structure to use "Type"
DEFAULT_MODULES = {
  "modules": [
    {
      "Type": "ปัจจัยตัวอย่าง",
      "submodules": [
        {
          "name": "หัวข้อย่อยตัวอย่าง",
          "questions": ["คำถามตัวอย่าง 1"]
        }
      ]
    }
  ]
}

if not os.path.exists(MODULES_FILE):
    with open(MODULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_MODULES, f, ensure_ascii=False, indent=2)

with open(MODULES_FILE, 'r', encoding='utf-8') as f:
    ALL_MODULES = json.load(f)['modules']

# ------------------ Helpers ------------------
# (tts_iapp and stt_whisper functions remain exactly the same)
def tts_iapp(text: str):
    output_filename = f"{uuid.uuid4()}.wav"
    output_filepath = os.path.join(CACHE_DIR, output_filename)
    form_data = {'text': (None, text), 'language': (None, "TH")}
    headers = {'apikey': API_KEY, 'User-Agent': 'python-requests/2.x'}
    try:
        response = requests.post(TTS_URL, files=form_data, headers=headers)
        if response.status_code == 200:
            with open(output_filepath, 'wb') as f: f.write(response.content)
            return url_for('static', filename=f'audio/{output_filename}')
    except Exception as e:
        print(f"An error occurred during TTS request: {e}")
        return None

def stt_whisper(filename, model_name=WHISPER_MODEL):
    if whisper is None: raise RuntimeError('whisper python package not installed')
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(filename, language='th', task='transcribe')
        return result.get('text','').strip()
    except Exception as e:
        print(f"Error during Whisper STT: {e}")
        return ""

# ------------------ NEW: Dialogue Manager for 1 Module ------------------
def get_next_robot_turn(session):
    """
    Handles conversation flow by selecting 1 main module and iterating through its submodules.
    """
    state = session.get('state', 'start')
    
    if state == 'start':
        session['state'] = 'awaiting_permission'
        return 'สวัสดีครับ ผมขอรบกวนถามคำถามสั้นๆ ได้ไหมครับ'
    
    if state == 'permission_denied':
        session['state'] = 'end'
        return 'ขอบคุณครับ แล้วขอให้มีวันที่ดีนะครับ'

    if state == 'ask_daily_activity':
        session['state'] = 'awaiting_activity'
        return 'วันนี้ท่านมาใช้บริการอะไรที่โรงพยาบาลครับ'

    # This is the starting point for the main conversation flow
    if state == 'awaiting_activity':
        # 1. Randomly choose ONE main module
        chosen_main_module = random.choice(ALL_MODULES)
        
        # 2. Store the plan in the session
        # CHANGED: Accessing 'Type' instead of 'name'
        session['main_module_name'] = chosen_main_module['Type']
        session['submodules'] = chosen_main_module['submodules']
        session['submodule_index'] = 0
        
        session['state'] = 'asking_questions'
        # 3. Return an introductory message for the chosen module
        return f"ขออนุญาติสอบถามเกี่ยวกับ'{session['main_module_name']}' สั้นๆ นะครับ"

    if state == 'asking_questions':
        submodule_index = session.get('submodule_index', 0)
        submodules_list = session.get('submodules', [])
        
        # Check if we have asked about all submodules
        if submodule_index < len(submodules_list):
            current_submodule = submodules_list[submodule_index]
            question_to_ask = random.choice(current_submodule['questions'])
            session['submodule_index'] += 1
            return question_to_ask
        else:
            # We've finished all submodules
            session['state'] = 'end'
            return 'ขอบคุณมากสำหรับคำตอบของท่าน ขอให้มีวันที่ดีนะครับ'
            
    return None # Fallback

# ------------------ Flask Web Routes ------------------
# (No major changes in the route logic itself)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_conversation():
    session_id = str(uuid.uuid4())
    sessions[session_id] = { 'session_id': session_id, 'started_at': datetime.utcnow().isoformat(), 'turns': [], 'state': 'start' }
    robot_text = get_next_robot_turn(sessions[session_id])
    audio_url = tts_iapp(robot_text)
    sessions[session_id]['turns'].append({'who':'robot','text':robot_text})
    return jsonify({'session_id': session_id, 'robot_text': robot_text, 'audio_url': audio_url})

@app.route('/chat', methods=['POST'])
def handle_chat():
    session_id = request.form.get('session_id')
    if not session_id or session_id not in sessions: return jsonify({'error': 'Invalid session ID'}), 400
    session = sessions[session_id]
    
    audio_file = request.files['audio']
    filepath = f"{AUDIO_FILENAME_PREFIX}{session_id}.wav"
    audio_file.save(filepath)
    user_text = stt_whisper(filepath)
    session['turns'].append({'who':'patient', 'text_raw': user_text})
    os.remove(filepath)

    current_state = session['state']
    if current_state == 'awaiting_permission':
        if any(w in user_text for w in ['ไม่','ไม่สะดวก','ไม่เอา']):
            session['state'] = 'permission_denied'
        else:
            session['state'] = 'ask_daily_activity'
    
    robot_text = get_next_robot_turn(session)
    if robot_text: session['turns'].append({'who': 'robot', 'text': robot_text})
    audio_url = tts_iapp(robot_text) if robot_text else None
    is_finished = (session.get('state') == 'end')
    
    response_data = { 'user_text': user_text, 'robot_text': robot_text, 'audio_url': audio_url, 'finished': is_finished }

    # 6. If the conversation is finished, save the log and clean up the session
    if is_finished:
        log_filename = os.path.join(OUTPUT_DIR, f'chatlog-{session_id}.json')
        session['ended_at'] = datetime.utcnow().isoformat()
        
        print(f"Conversation finished. Saving log to {log_filename}") # Added a print statement for debugging
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
        
        del sessions[session_id] # Clean up session from memory

    return jsonify(response_data)

# ------------------ NEW: Form Logic for 1 Module ------------------
@app.route('/get_form_questions', methods=['GET'])
def get_form_questions():
    """
    Randomly selects ONE main module and returns ONE random question from EACH of its submodules.
    """
    # 1. Randomly choose ONE main module
    chosen_main_module = random.choice(ALL_MODULES)

    # 2. Collect ONE random question from EACH submodule of the chosen main module
    all_questions = []
    for submodule in chosen_main_module.get('submodules', []):
        if submodule.get('questions'):
            question_to_add = random.choice(submodule['questions'])
            all_questions.append(question_to_add)
            
    return jsonify({'questions': all_questions})

@app.route('/submit_form', methods=['POST'])
def submit_form():
    form_data = request.form.to_dict()
    form_id = str(uuid.uuid4())
    log_filename = os.path.join(OUTPUT_DIR, f'formlog-{form_id}.json')
    output_log = { "form_id": form_id, "submitted_at": datetime.utcnow().isoformat(), "responses": form_data }
    print(f"Form submitted. Saving log to {log_filename}")
    with open(log_filename, 'w', encoding='utf-8') as f: json.dump(output_log, f, ensure_ascii=False, indent=2)
    return jsonify({'status': 'success', 'message': 'Form submitted successfully!'})

# ------------------ MAIN ------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)