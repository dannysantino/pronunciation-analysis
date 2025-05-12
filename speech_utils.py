import os
import asyncio
import random

from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION")

async def send_to_openai(openai_ws, base64_audio):
    audio_byte = {
        "type": "input_audio_buffer.append",
        "audio": base64_audio
    }
    await openai_ws.send_json(audio_byte)

async def update_default_session(openai_ws):
    system_message = (
        "You are a helpful language learning voice assistant. "
        "Your job is to engage the caller in an interesting conversation to help them improve their pronunciation. "
        "If the caller speaks in a non-English language, you should respond in a standard accent or dialect they might find familiar. "
        "Keep your responses as short as possible."
    )

    voices = [
        "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"
    ]
    voice = random.choice(voices)
    print(voice)

    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": system_message,
            "voice": voice,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": { "type": "server_vad" },
            "temperature": 0.8
        }
    }

    await openai_ws.send_json(session_update)

class AzureSpeechRecognizer():
    def __init__(self):
        self.results = []
    def configure(self, language):
        self.language = language
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)

        # set up audio input stream
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=8000,
            bits_per_sample=16,
            channels=1,
            wave_stream_format=speechsdk.AudioStreamWaveFormat.MULAW
        )
        self.stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self.stream)

        # instantiate the speech recognizer
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)
        self.recognition_done = asyncio.Event()
        
        # configure pronunciation assessment
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
        )
        # prosody scores only available if language/locale == "en-US":
        if language == "en-US":
            pronunciation_config.enable_prosody_assessment()
        pronunciation_config.apply_to(self.speech_recognizer)
        print("Azure speech recognizer configured:", language)

        # define callbacks to signal events fired by the speech recognizer
        self.speech_recognizer.recognizing.connect(lambda evt: print(f"Recognizing: {evt.result.text}"))
        self.speech_recognizer.recognized.connect(self.recognized_cb)
        self.speech_recognizer.session_started.connect(lambda evt: print(f"Azure session started: {evt.session_id}"))
        self.speech_recognizer.session_stopped.connect(self.stopped_cb)
        self.speech_recognizer.canceled.connect(self.canceled_cb)

    def start_recognition(self):
        result_future = self.speech_recognizer.start_continuous_recognition_async()
        result_future.get()

    def stop_recognition(self):
        self.speech_recognizer.stop_continuous_recognition_async()

    async def send_to_azure(self, audio):
        try:
            self.stream.write(audio)
        except Exception as e:
            print(f"Error writing to Azure stream: {e}")

    # callback if speech is recognized
    def recognized_cb(self, evt: speechsdk.RecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"\nPronunciation assessment for: {evt.result.text}")
            pronunciation_result = speechsdk.PronunciationAssessmentResult(evt.result)
            print(
                f"Accuracy: {pronunciation_result.accuracy_score} \n\n"
                f"Pronunciation score: {pronunciation_result.pronunciation_score} \n\n"
                f"Completeness score: {pronunciation_result.completeness_score} \n\n"
                f"Fluency score: {pronunciation_result.fluency_score}\n\n"
                f"Prosody score: {pronunciation_result.prosody_score} \n\n"
            )

            # provide further analysis
            print("     Word-level details:")
            for idx, word in enumerate(pronunciation_result.words):
                print(f"     {idx + 1}. word: {word.word}\taccuracy: {word.accuracy_score}\terror type: {word.error_type}")
            
            # gather results to send to caller
            analysis = {
                "Assessment for:": evt.result.text,
                "Accuracy -": pronunciation_result.accuracy_score,
                "Pronunciation -": pronunciation_result.pronunciation_score,
                "Completeness -": pronunciation_result.completeness_score,
                "Fluency -": pronunciation_result.fluency_score
            }
            if self.language == "en-US":
                analysis["Prosody -"] = pronunciation_result.prosody_score
            self.results.append(analysis)

    def canceled_cb(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            print(f"Cancellation details: {cancellation_details.reason.CancellationReason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                self.recognition_done.set()

    def stopped_cb(self, evt):
        print(f"Azure session stopped: {evt.session_id}")
        self.recognition_done.set()