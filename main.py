import os
import asyncio
import aiohttp
import json
import base64

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from dotenv import load_dotenv
import uvicorn
from twilio.twiml.voice_response import VoiceResponse, Connect

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VOICE_AGENT_INSTRUCTIONS = """
You are a helpful language learning voice assistant.
Your job is to engage the user in an interesting conversation to help them improve their pronunciation.
If the user speaks in a non-English language, you should respond in a standard accent or dialect they might find familiar.
Keep your responses as short as possible.
"""

@app.api_route("/voice", methods=["GET", "POST"])
async def voice_response(request: Request):
    response = VoiceResponse()
    response.say("Welcome to the AI Voice Assistant.")
    response.pause(length=1)
    response.say("You may begin speaking in your preferred language.")
    host = request.url.hostname
    print("host:", host)
    connect = Connect()
    connect.stream(url=f"wss://{host}/audio-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/audio-stream")
async def stream_audio(twilio_ws: WebSocket):
    await twilio_ws.accept()
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                url,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            ) as openai_ws:
                stream_sid = None
                
                await update_default_session(openai_ws)

                async def receive_twilio_stream():
                    nonlocal stream_sid
                    
                    try:
                        async for message in twilio_ws.iter_text():
                            data = json.loads(message)

                            match data["event"]:
                                case "connected":
                                    print("Connected to Twilio media stream")
                                case "start":
                                    print("Twilio stream has started")
                                    stream_sid = data["start"]["streamSid"]
                                    print("Stream SID:", stream_sid)
                                case "media":
                                    base64_audio = data["media"]["payload"]
                                    # audio_byte = {
                                    #     "type": "input_audio_buffer.append",
                                    #     "audio": base64_audio
                                    # }
                                    # await openai_ws.send_json(audio_byte)
                                    openai_task = asyncio.create_task(send_to_openai(base64_audio, openai_ws))
                                    # azure_task = asyncio.create_task(send_to_azure(base64_audio))
                                    await asyncio.gather(openai_task)
                                case "stop":
                                    await openai_ws.close()
                    except WebSocketDisconnect:
                        if not openai_ws.closed:
                            await openai_ws.close()

                async def send_ai_response():
                    nonlocal stream_sid

                    try:
                        async for ws_message in openai_ws:
                            openai_response = json.loads(ws_message.data)

                            if ws_message.type == aiohttp.WSMsgType.TEXT:
                                print("wsmessage.text: \n\n", openai_response)
                                if openai_response["type"] == "error":
                                    print("Error in OpenAI response:", openai_response["error"])
                                if openai_response["type"] == "input_audio_buffer.speech_started":
                                    print("Speech started")
                                    await twilio_ws.send_json({
                                        "event": "clear",
                                        "streamSid": stream_sid
                                    })
                            if ws_message.type == aiohttp.WSMsgType.BINARY:
                                print("wsmessage.binary: \n")

                                if openai_response["type"] == "response.audio.delta":
                                        print("ai audio response:", openai_response["delta"])
                                        audio_payload = base64.b64encode(base64.b64decode(openai_response["delta"])).decode("utf-8")
                                        audio_data = {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {
                                                "payload": audio_payload
                                            }
                                        }

                                        await twilio_ws.send_json(audio_data)
                                        print("Audio data sent to Twilio WebSocket")

                                        mark_message = {
                                            "event": "mark",
                                            "streamSid": stream_sid,
                                            "mark": { "name": "ai response" }
                                        }

                                        await twilio_ws.send_json(mark_message)
                                        print("Mark message sent to Twilio WebSocket")
                            if openai_response["type"] == "response.audio.delta":
                                print("ai audo response recieved.")
                    except Exception as e:
                        print(f"Error in send_ai_response: {e}")

                await asyncio.gather(receive_twilio_stream(), send_ai_response())
    except Exception as e:
        print("Error in aiohttp Websocket connection:", e)
        await twilio_ws.close()

async def send_to_openai(base64_audio, openai_ws):
    audio_byte = {
        "type": "input_audio_buffer.append",
        "audio": base64_audio
    }
    await openai_ws.send_json(audio_byte)

async def send_to_azure(base64_audio):
    print("Sending audio to Azure")

async def update_default_session(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": VOICE_AGENT_INSTRUCTIONS,
            "voice": "ash",
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": { "type": "server_vad" },
            "temperature": 0.8
        }
    }

    await openai_ws.send_json(session_update)
    print("Sent OpenAI session update")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



# @app.websocket("/stream")
# async def stream_audio(websocket: WebSocket):
#     print("reached websocket endpoint")
#     await websocket.accept()
#     print("WebSocket connection accepted")

#     try:
#         async with aiohttp.ClientSession() as session:
#             openai_ws = await session.ws_connect(
#                 "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
#                 headers={
#                     "Authorization": f"Bearer { OPENAI_API_KEY }",
#                     "OpenAI-Beta": "realtime=v1"
#                 }
#             )
#             print("Connected to OpenAI WebSocket")
#             openai_session = await update_default_session(openai_ws)
#             print("Session initialized with OpenAI WebSocket:", openai_session)

#             openai_task = asyncio.create_task(handle_openai_stream(websocket, openai_ws))
#             # azure_task = asyncio.create_task(handle_azure_stream(websocket))
#             await asyncio.gather(openai_task)
#     except Exception as e:
#         print(f"Error in aiohttp WebSocket connection: {e}")

# async def handle_openai_stream(twilio_ws: WebSocket, openai_ws: aiohttp.ClientWebSocketResponse):
#     stream_sid = None

#     async def send_to_openai():
#         nonlocal stream_sid
#         try:
#             async for message in twilio_ws.iter_text():
#                 data = json.loads(message)

#                 if data["event"] == "connected":
#                     print("Connected to Twilio WebSocket")
#                 if data["event"] == "start":
#                     stream_sid = data["start"]["streamSid"]
#                     print("Twilio stream has started.")
#                     print("Stream SID:", stream_sid)
#                 elif data["event"] == "media" and not openai_ws.closed:
#                     audio_byte = {
#                         "type": "input_audio_buffer.append",
#                         "audio": data["media"]["payload"]
#                     }
#                     await openai_ws.send_json(audio_byte)
#                     # print("Audio data sent to OpenAI WebSocket")
#                 elif data["event"] == "stop":
#                     await openai_ws.close()
#         except WebSocketDisconnect:
#             if not openai_ws.closed:
#                 await openai_ws.close()
#             print("Error in handle_openai_stream")

#     async def send_ai_response():
#         nonlocal stream_sid
#         try:
#             async for ws_message in openai_ws:
#                 openai_response = json.loads(ws_message.data)
#                 # print("open ai response type: \n", openai_response["type"])

#                 if ws_message.type == aiohttp.WSMsgType.TEXT:
#                     print("wsmessage.text: \n\n", openai_response)
#                     if openai_response["type"] == "error":
#                         print("Error in OpenAI response:", openai_response["error"])
#                 elif ws_message.type == aiohttp.WSMsgType.BINARY:
#                     print("wsmessage.binary: \n")
#                     if openai_response["type"] == "response.audio.delta":
#                             print("ai audio response:", openai_response["delta"])
#                             audio_payload = base64.b64encode(base64.b64decode(openai_response["delta"])).decode("utf-8")
#                             audio_data = {
#                                 "event": "media",
#                                 "streamSid": stream_sid,
#                                 "media": {
#                                     "payload": audio_payload
#                                 }
#                             }

#                             await twilio_ws.send_bytes(audio_data)
#                             print("Audio data sent to Twilio WebSocket")

#                             mark_message = {
#                                 "event": "mark",
#                                 "streamSid": stream_sid,
#                                 "mark": { "name": "ai response" }
#                             }
#                             await twilio_ws.send_json(mark_message)
#                             print("Mark message sent to Twilio WebSocket")
#                 elif openai_response["type"] == "response.audio.delta":
#                     print("ai audo response recieved.")
#         except Exception as e:
#             print(f"Error in send_ai_response: {e}")

#     # Start both tasks concurrently
#     await asyncio.gather(send_to_openai(), send_ai_response())


# @app.websocket("/stream")
# async def stream(websocket: WebSocket):
#     print("reached stream websocket")
#     await websocket.accept()

#     try:
#         async with aiohttp.ClientSession() as session:
#             openai_ws = await session.ws_connect(
#                 "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
#                 headers={
#                     "Authorization": f"Bearer { OPENAI_API_KEY }",
#                     "OpenAI-Beta": "realtime=v1"
#                 }
#             )
#             print("Connected to OpenAI WebSocket")
#             openai_session = await initialize_realtime_session(openai_ws)
#             print("Session initialized with OpenAI WebSocket:", openai_session)

#             await asyncio.gather(send_to_oai(websocket, openai_ws), send_to_twilio(websocket, openai_ws))
#     except Exception as e:
#         print(f"Error in aiohttp WebSocket connection: {e}")

# async def send_to_oai(twilio_ws: WebSocket, openai_ws: aiohttp.ClientWebSocketResponse):
#     stream_sid = None
#     try:
#         async for message in twilio_ws.iter_text():
#             data = json.loads(message)

#             if data["event"] == "connected":
#                 print("Connected to Twilio WebSocket")
#             elif data["event"] == "start":
#                 stream_sid = data["start"]["streamSid"]
#                 print("Twilio stream has started.")
#                 print("Stream SID:", stream_sid)
#             elif data["event"] == "media":
#                 audio_byte = {
#                     "type": "input_audio_buffer.append",
#                     "audio": data["media"]["payload"]
#                 }
#                 # await openai_ws.send_json(audio_byte)
#                 # print("Audio data sent to OpenAI WebSocket")
#             elif data["event"] == "stop":
#                 await openai_ws.close()
#     except WebSocketDisconnect:
#         print(f"Error in twilio WebSocket connection")
#         await twilio_ws.close()
        
# async def send_to_twilio(twilio_ws: WebSocket, openai_ws: aiohttp.ClientWebSocketResponse):
#     stream_sid = None
#     try:
#         response = await openai_ws.send_json({
#             "event_id": "ai_response_24",
#             "type": "conversation.item.retrieve",
#             "item_id": "item_BPV7c8zjL304b4McyDBDR"
#         })
#         print("sent retrieval request to open_ai")

#         print("response type:", type(response))
#         print("openai response \n\n:", response)
#         ai_response = json.loads(response)
#         print("Received AI response:", ai_response)

#         audio = ai_response["item"]["content"][0]["audio"]

#         audio_payload = base64.b64encode(base64.b64decode(audio["delta"])).decode("utf-8")
#         audio_data = {
#             "event": "media",
#             "streamSid": stream_sid,
#             "media": {
#                 "payload": audio_payload
#             }
#         }
#         await twilio_ws.send_json(audio_data)
#         print("Audio data sent to Twilio WebSocket")

#         mark_message = {
#             "event": "mark",
#             "streamSid": stream_sid,
#             "mark": { "name": "ai response" }
#         }
#         await twilio_ws.send_json(mark_message)
#         print("Mark message sent to Twilio WebSocket")
#     except Exception as e:
#         print(f"Error in send_ai_response: {e}")