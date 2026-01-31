# Chronicle Backend

AI-powered journaling app with speech-to-text, CBT-style chat, and text-to-speech capabilities. Built with FastAPI and cloud-based AI models.

## Features

- **Speech-to-Text (ASR)**: Convert audio recordings to text using Whisper
- **CBT-Style Chat**: Get supportive, reflective responses using LangChain
- **Text-to-Speech (TTS)**: Convert AI responses to audio
- **Conversation Memory**: Maintains context across chat interactions
- **Cloud-First**: All heavy AI processing is handled by cloud APIs

## Architecture

This backend follows production best practices:

- **Dependency Management**: PDM for modern Python dependency management
- **Configuration**: Pydantic Settings for type-safe environment-based config
- **Lifecycle Management**: Proper startup/shutdown events for resource initialization
- **Dependency Injection**: FastAPI dependencies for shared resources
- **Structured Logging**: Environment-aware logging with color support in development
- **Error Handling**: Global exception handlers with proper HTTP status codes
- **Modular Structure**: Clean separation of concerns (config, utils, endpoints)

## Project Structure

```
audiodiary/
├── app.py                 # Main FastAPI application
├── config/                # Configuration management
│   ├── settings.py       # Pydantic settings
│   └── logging.py        # Logging configuration
├── utils/                 # Utility functions
│   ├── dependencies.py   # Dependency injection
│   ├── asr_utils.py      # ASR processing
│   ├── chat_utils.py     # LangChain chat logic
│   └── tts_utils.py      # TTS processing
├── api_v1/               # API endpoints
│   ├── api.py           # Router aggregation
│   ├── schemas.py       # Pydantic models
│   └── endpoints/       # Individual endpoints
│       ├── asr_endpoint.py
│       ├── health_chat.py
│       ├── tts_endpoint.py
│       └── health.py
├── pyproject.toml        # PDM dependencies
└── .env                  # Environment variables (not in git)
```

## Setup

### Prerequisites

- Python 3.11+
- PDM (Python Dependency Manager)
- HuggingFace API token
- LLM API key (OpenAI, Anthropic, etc.)

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/audiodiary
   ```

2. **Install PDM** (if not already installed)
   ```bash
   curl -sSL https://pdm-project.org/install-pdm.py | python3 -
   ```

3. **Install dependencies**
   ```bash
   pdm install
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

   Required environment variables:
   - `HUGGINGFACE_API_KEY`: Your HuggingFace API token
   - `LLM_API_KEY`: Your LLM provider API key
   - `LLM_PROVIDER`: LLM provider (openai, anthropic, etc.)

### Running the Server

**Development mode** (with auto-reload):
```bash
pdm run dev
```

**Production mode**:
```bash
pdm run start
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```
Returns service status and dependency health.

### ASR (Speech-to-Text)
```bash
POST /api/v1/asr
Content-Type: multipart/form-data

file: <audio file (WAV, MP3, M4A)>
```
Returns transcribed text.

### Chat (CBT-Style Reflection)
```bash
POST /api/v1/chat
Content-Type: application/json

{
  "message": "I had a really stressful day at work today.",
  "user_id": "optional_user_id"
}
```
Returns supportive AI response.

### TTS (Text-to-Speech)
```bash
POST /api/v1/tts
Content-Type: application/json

{
  "text": "Your AI response text here"
}
```
Returns audio file (WAV format).

## Example Usage

### Complete Flow

1. **Upload audio**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/asr \
     -F "file=@recording.wav" \
     -o transcription.json
   ```

2. **Get AI reflection**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "I had a stressful day"}' \
     -o response.json
   ```

3. **Convert response to audio**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "It sounds like today was challenging..."}' \
     --output response.wav
   ```

## Configuration

All configuration is managed via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_API_KEY` | - | HuggingFace API token (required) |
| `LLM_API_KEY` | - | LLM provider API key (required) |
| `LLM_PROVIDER` | `openai` | LLM provider (openai/anthropic/etc.) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | ASR model |
| `TTS_MODEL` | `facebook/mms-tts-eng` | TTS model |
| `ENVIRONMENT` | `development` | Environment (development/production) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CONVERSATION_MEMORY_SIZE` | `5` | Number of messages to keep in context |
| `MAX_AUDIO_FILE_SIZE_MB` | `25` | Maximum audio upload size |

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Fast linting

Run formatters:
```bash
pdm run black .
pdm run ruff check .
```

### Testing

```bash
pdm run pytest
```

## Production Deployment

1. Set `ENVIRONMENT=production` in `.env`
2. Set `DEBUG=false`
3. Configure appropriate `CORS_ORIGINS`
4. Use a production WSGI server or run with uvicorn workers:
   ```bash
   pdm run uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## License

MIT

## Disclaimer

**This is a journaling assistant, not a medical service.** It provides supportive reflections based on CBT principles but does not offer medical advice, diagnosis, or treatment.
