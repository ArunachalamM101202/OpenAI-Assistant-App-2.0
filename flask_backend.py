
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import openai
from openai import OpenAI
import time
from dotenv import load_dotenv
import os
from typing import IO
import pandas as pd
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_ASSISTANT_ID = os.getenv("ASSISTANT_ID")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

app = Flask(__name__)

# Initialize OpenAI and ElevenLabs clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# def text_to_speech_stream(text: str) -> IO[bytes]:
#     response = elevenlabs_client.text_to_speech.convert(
#         voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
#         output_format="mp3_22050_32",
#         text=text,
#         model_id="eleven_multilingual_v2",
#         voice_settings=VoiceSettings(
#             stability=0.0,
#             similarity_boost=1.0,
#             style=0.0,
#             use_speaker_boost=True,
#         ),
#     )

#     audio_stream = BytesIO()
#     for chunk in response:
#         if chunk:
#             audio_stream.write(chunk)
#     audio_stream.seek(0)
#     return audio_stream

def text_to_speech_stream(text: str) -> IO[bytes]:
    response = elevenlabs_client.text_to_speech.convert(
        voice_id=chat_session.current_voice_id,  # Use the selected voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    return audio_stream


class ChatSession:
    def __init__(self):
        self.thread_id = None
        self.assistant_id = DEFAULT_ASSISTANT_ID
        self.messages = []
        self.temperature = 0.7
        self.current_voice_id = "pNInz6obpgDQGcFmaJgB"  # default voice

    def create_thread(self):
        self.thread_id = openai_client.beta.threads.create().id
        self.messages = []
        return self.thread_id

    def process_message(self, user_message):
        if not self.thread_id:
            self.create_thread()

        # Send message to OpenAI
        openai_client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=user_message
        )

        # Create and monitor run
        run = openai_client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            temperature=self.temperature
        )

        while True:
            run_status = openai_client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status == 'failed':
                return {"error": "Assistant response failed"}
            time.sleep(1)

        # Get messages
        messages = openai_client.beta.threads.messages.list(
            thread_id=self.thread_id)
        return messages.data


# Create global chat session
chat_session = ChatSession()


@app.route('/')
def home():
    return render_template('index.html')


# Add this new route to get voices
@app.route('/get_voices')
def get_voices():
    try:
        # Read the CSV file
        voices_df = pd.read_csv(
            'voices.csv', header=None, names=['name', 'id'])
        # Convert to list of dictionaries
        voices = voices_df.to_dict('records')
        return jsonify(voices)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this new route to update voice


# @app.route('/update_voice', methods=['POST'])
# def update_voice():
#     data = request.json
#     new_voice_id = data.get('voice_id')

#     if not new_voice_id:
#         return jsonify({"error": "No voice ID provided"}), 400

#     chat_session.current_voice_id = new_voice_id
#     return jsonify({"success": True, "message": "Voice updated"})

@app.route('/update_voice', methods=['POST'])
def update_voice():
    data = request.json
    new_voice_id = data.get('voice_id')

    if not new_voice_id:
        return jsonify({"error": "No voice ID provided"}), 400

    chat_session.current_voice_id = new_voice_id
    return jsonify({
        "success": True,
        "message": "Voice updated",
        "voice_id": new_voice_id
    })


@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    messages = chat_session.process_message(user_message)

    if isinstance(messages, dict) and "error" in messages:
        return jsonify(messages), 500

    response_messages = []
    assistant_audio_id = None

    for msg in reversed(messages):
        message_data = {
            "role": msg.role,
            "content": msg.content[0].text.value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Generate audio for assistant messages
        if msg.role == "assistant":
            try:
                audio_stream = text_to_speech_stream(msg.content[0].text.value)
                assistant_audio_id = datetime.now().strftime("%Y%m%d%H%M%S")
                # Store audio stream in session
                chat_session.audio_streams = getattr(
                    chat_session, 'audio_streams', {})
                chat_session.audio_streams[assistant_audio_id] = audio_stream
                message_data["audio_id"] = assistant_audio_id
            except Exception as e:
                print(f"Error generating audio: {e}")

        response_messages.append(message_data)

    return jsonify({"messages": response_messages})


@app.route('/get_audio/<audio_id>')
def get_audio(audio_id):
    if hasattr(chat_session, 'audio_streams') and audio_id in chat_session.audio_streams:
        audio_stream = chat_session.audio_streams[audio_id]
        audio_stream.seek(0)
        return send_file(
            audio_stream,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=f'response_{audio_id}.mp3'
        )
    return jsonify({"error": "Audio not found"}), 404


@app.route('/get_current_voice')
def get_current_voice():
    return jsonify({"voice_id": chat_session.current_voice_id})


@app.route('/update_assistant', methods=['POST'])
def update_assistant():
    data = request.json
    new_assistant_id = data.get('assistant_id')

    if not new_assistant_id:
        return jsonify({"error": "No assistant ID provided"}), 400

    chat_session.assistant_id = new_assistant_id
    chat_session.create_thread()

    return jsonify({"success": True, "message": "Assistant ID updated"})


@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_file.save("temp_audio.mp3")  # Save temporarily

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        with open("temp_audio.mp3", "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )

        return jsonify({"text": transcription.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.getenv("PORT", 10000))  # Get PORT from Render, default 10000
    app.run(host='0.0.0.0', port=port, debug=False)
