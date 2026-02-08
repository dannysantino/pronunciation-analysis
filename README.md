# Pronunciation Analysis

An AI voice coach that lets callers practice speaking in multiple languages and receive real-time pronunciation assessment. It uses Twilio Programmable Voice for calls, OpenAI's Realtime API for live speech-to-speech responses, and Azure Speech for pronunciation scoring. After a call, the app sends a summary of the assessment via WhatsApp.

> [!NOTE]
>
> `ARTICLE.md` is the technical article written for the Twilio blog. You can also read the published version [here](https://www.twilio.com/en-us/blog/ai-voice-analyze-pronunciation-twilio-programmable-voice-openai-azure-speech)

## Features

- Real-time, multi-language voice practice over a Twilio phone number.
- Speech-to-speech responses powered by the OpenAI Realtime API.
- Pronunciation assessment (accuracy, pronunciation, completeness, fluency, and prosody for en-US) via Azure Speech.
- Post-call WhatsApp summary delivered through Twilio Messaging.

## How It Works

1. A caller dials your Twilio number and selects a language.
2. Twilio streams the audio to your server via Media Streams.
3. The server forwards audio to OpenAI (for responses) and Azure (for assessment) in parallel.
4. OpenAI audio responses are streamed back to the caller.
5. When the call ends, the server sends a WhatsApp message with the assessment results.

## Project Structure

- `main.py` - FastAPI server, Twilio call flow, WebSocket streaming, and WhatsApp delivery.
- `speech_utils.py` - Azure Speech recognizer and OpenAI session helpers.
- `requirements.txt` - Python dependencies.
- `ARTICLE.md` - Full tutorial for the project.

## Requirements

- Python 3.12+ (the tutorial uses 3.12.4).
- Twilio account with:
  - A voice-enabled phone number.
  - WhatsApp Sandbox enabled.
- OpenAI API key with Realtime API access.
- Azure account with a Speech resource.
- ngrok account and authtoken.

## Setup

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure services

Follow the steps in `ARTICLE.md` to set up:

- Azure Speech resource (collect key + region).
- Twilio Voice number (collect Account SID, Auth Token, and Number SID).
- Twilio WhatsApp Sandbox (join the sandbox and collect From/To numbers).
- OpenAI API key.
- ngrok authtoken and optional static domain.

### 3) Create a `.env` file

```text
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_NUMBER_SID=...
TWILIO_SANDBOX_NUMBER=...
WHATSAPP_PHONE_NUMBER=...

OPENAI_API_KEY=...

AZURE_SPEECH_KEY=...
AZURE_SERVICE_REGION=...

NGROK_AUTHTOKEN=...
```

### 4) Update ngrok configuration (optional but recommended)

`main.py` uses a reserved ngrok domain in the FastAPI lifespan event:

```python
listener = await ngrok.forward(
    addr=PORT,
    proto="http",
    domain="select-shining-coral.ngrok-free.app"
)
```

- Replace `select-shining-coral.ngrok-free.app` with your reserved domain.
- If you are not using a reserved domain, remove the `domain` line so ngrok generates a random URL.

When the app starts, it automatically updates the Twilio voice webhook to `https://<ngrok-domain>/gather`.

## Run the App

```bash
python main.py
```

Call your Twilio number and follow the voice prompts. When you hang up, a WhatsApp message will deliver your pronunciation assessment.

## Configuration Notes

- Language selection and prompts live in `language_mapping` in `main.py`.
- The OpenAI model is set to `gpt-4o-mini-realtime-preview-2024-12-17` in `main.py`.
- Prosody scores are only available when `language == "en-US"` in `speech_utils.py`.

## Troubleshooting

- Twilio trial accounts have limitations. Check Twilio's trial restrictions if calls or WhatsApp messages fail.
- If you get OpenAI responses but no Azure output, temporarily comment out the OpenAI task to isolate the Azure stream.
- Ensure your ngrok tunnel is live and the Twilio webhook points to `/gather`.

## License

See `LICENSE`.
