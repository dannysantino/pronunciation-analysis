import os
import asyncio
import aiohttp
import json
import base64

from dotenv import load_dotenv
import ngrok
import uvicorn

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather, Connect

from speech_utils import AzureSpeechRecognizer, send_to_openai, update_default_session

load_dotenv(override=True)

app = FastAPI()

PORT = 5000

NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER_SID = os.getenv("TWILIO_NUMBER_SID")
TWILIO_SANDBOX_NUMBER = os.getenv("TWILIO_SANDBOX_NUMBER")
WHATSAPP_PHONE_NUMBER = os.getenv("WHATSAPP_PHONE_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

speech_recognizer = AzureSpeechRecognizer()

language_mapping = {
    "1": {
        "code": "en-US",
        "initial_prompt": "Starting the interactive session... What would you like to talk about today?"
    },
    "2": {
        "code": "es-MX",
        "initial_prompt": "Iniciando la sesión interactiva... ¿De qué le gustaría hablar hoy?"
    },
    "3": {
        "code": "fr-FR",
        "initial_prompt": "Démarrage de la session interactive... De quoi aimeriez-vous parler aujourd'hui?"
    }
}

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Setting up ngrok tunnel...")
    ngrok.set_auth_token(NGROK_AUTHTOKEN)

    # establish connectivity
    listener = await ngrok.forward(
        addr=PORT,
        proto="http",
        domain="select-shining-coral.ngrok-free.app"
    )
    print(listener.url())

    # configure twilio webhook to use the generated url
    twilio_phone = client.incoming_phone_numbers(
        TWILIO_NUMBER_SID
    ).update(voice_url=listener.url() + "/gather")
    print("Twilio voice URL updated:", twilio_phone.voice_url)
    
    # error handling and cleanup
    try:
        yield
    except asyncio.CancelledError:
        print("Lifespan cancel received.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt in lifespan")
    finally:
        print("Tearing down ngrok tunnel...")
        ngrok.disconnect()

app = FastAPI(lifespan=lifespan)


@app.api_route("/gather", methods=["GET", "POST"])
def gather():
    response = VoiceResponse()
    gather = Gather(num_digits=1, action="/voice")
    gather.say("Welcome to the Language Assistant. For English, press 1.")
    gather.say("Para español, presione 2.", language="es-MX")
    gather.say("Pour le français, appuyez sur 3.", language="fr-FR")
    response.append(gather)

    # if caller fails to select an option, redirect them into a loop
    response.redirect("/gather")
    
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.api_route("/voice", methods=["GET", "POST"])
async def voice_response(request: Request):
    response = VoiceResponse()
    form_data = await request.form()

    if "Digits" in form_data:
        choice = form_data["Digits"]
        if choice in language_mapping:
            language = language_mapping.get(choice, {}).get("code")
            prompt = language_mapping.get(choice, {}).get("initial_prompt")
            response.say(prompt, language=language)
            host = request.url.hostname
            connect = Connect()
            connect.stream(
                url=f"wss://{host}/audio-stream/{language}",
                status_callback=f"https://{host}/send-analysis"
            )
            response.append(connect)

            return HTMLResponse(content=str(response), media_type="application/xml")
        
    # if caller selected invalid choice, redirect them to /gather
    response.say("Sorry, you have made an invalid selection. Please choose a number between 1 and 3.")
    response.redirect("/gather")
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/audio-stream/{language}")
async def stream_audio(twilio_ws: WebSocket, language: str):
    await twilio_ws.accept()
    
    stream_sid = None
    speech_recognizer.configure(language)
    
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                url,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            ) as openai_ws:
                await update_default_session(openai_ws)

                # receive and process Twilio audio
                async def receive_twilio_stream():
                    nonlocal stream_sid

                    try:
                        async for message in twilio_ws.iter_text():
                            data = json.loads(message)

                            match data["event"]:
                                case "connected":
                                    print("Connected to Twilio media stream")
                                    speech_recognizer.start_recognition()
                                case "start":
                                    stream_sid = data["start"]["streamSid"]
                                    print("Twilio stream started:", stream_sid)
                                case "media":
                                    base64_audio = data["media"]["payload"]
                                    mulaw_audio = base64.b64decode(base64_audio)

                                    azure_task = asyncio.create_task(speech_recognizer.send_to_azure(mulaw_audio))
                                    openai_task = asyncio.create_task(send_to_openai(openai_ws, base64_audio))
                                    await asyncio.gather(azure_task, openai_task)
                                case "stop":
                                    print("Twilio stream has stopped")
                    except WebSocketDisconnect:
                        print("Twilio webSocket disconnected")
                    finally:
                        speech_recognizer.stream.close()
                        speech_recognizer.stop_recognition()
                        if not openai_ws.closed:
                            await openai_ws.close()

                # send AI response to Twilio
                async def send_ai_response():
                    nonlocal stream_sid

                    try:
                        async for ws_message in openai_ws:
                            openai_response = json.loads(ws_message.data)

                            if ws_message.type == aiohttp.WSMsgType.TEXT:
                                match openai_response["type"]:
                                    case "error":
                                        print("Error in OpenAI response:", openai_response["error"])
                                    case "input_audio_buffer.speech_started":
                                        print("Speech detected")
                                        await twilio_ws.send_json({
                                            "event": "clear",
                                            "streamSid": stream_sid
                                        })
                                    case "response.audio.delta":
                                        try:
                                            audio_payload = base64.b64encode(base64.b64decode(openai_response["delta"])).decode("utf-8")
                                            audio_data = {
                                                "event": "media",
                                                "streamSid": stream_sid,
                                                "media": {
                                                    "payload": audio_payload
                                                }
                                            }

                                            await twilio_ws.send_json(audio_data)

                                            # send mark message to signal media playback complete
                                            mark_message = {
                                                "event": "mark",
                                                "streamSid": stream_sid,
                                                "mark": { "name": "ai response" }
                                            }
                                            await twilio_ws.send_json(mark_message)
                                        except Exception as e:
                                            print("Error sending Twilio audio:", e)
                    except Exception as e:
                        print(f"Error in send_ai_response: {e}")

                await asyncio.gather(receive_twilio_stream(), send_ai_response())
    except Exception as e:
        print("Error in aiohttp Websocket connection:", e)
        await twilio_ws.close()

@app.api_route("/send-analysis", methods=["GET", "POST"])
async def send_analysis(request: Request):
    form_data = await request.form()
    stream_event = form_data["StreamEvent"]

    if stream_event == "stream-stopped":
        results = speech_recognizer.results

        if len(results) > 0:
            message_body = "\n----------------\n".join(
                "\n".join(f"{param} {result[param]}" for param in result)
                for result in results
            )
        else:
            message_body = "No assessment results available for your latest session."
            
        # call = client.calls(form_data["CallSid"]).fetch()
        message = client.messages.create(
            body=message_body,
            from_=f"whatsapp:{TWILIO_SANDBOX_NUMBER}",
            to=f"whatsapp:{WHATSAPP_PHONE_NUMBER}",
            status_callback=f"https://{request.url.hostname}/message-status"
        )
        print("Message SID:", message.sid)
    return "OK"

@app.api_route("/message-status", methods=["GET", "POST"])
async def message_status(request: Request):
    form_data = await request.form()
    print(f"Message status: {form_data["MessageStatus"]}")

    return "OK"


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=True)