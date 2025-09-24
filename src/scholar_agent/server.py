"""
FastAPI server that wraps the Scholar agent with an OpenAI-compatible API
"""

# Standard
from typing import Dict, List, Literal, Optional, Tuple
import asyncio
import ssl

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel

# Third Party
import aconfig
import alog
import uvicorn

# Local
from scholar_agent import config
from scholar_agent.scholar import ScholarAgentSession
from scholar_agent.utils.models import model_factory

# Set up logging
log = alog.use_channel("SERVER")

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)


# Define OpenAI-compatible request/response models
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class SessionInfo(BaseModel):
    """Information about a session"""

    id: str
    created: int
    last_used: int


class SessionListResponse(BaseModel):
    """Response model for listing sessions"""

    sessions: List[SessionInfo]


class ScholarServer:
    """FastAPI server that wraps the Scholar agent with an OpenAI-compatible API"""

    def __init__(self, config: aconfig.Config):
        self.config = config
        self.app = FastAPI(
            title="Scholar Agent API",
            description="OpenAI-compatible API for the Scholar Agent",
            version="0.1.0",
        )
        self.setup_routes()
        self.setup_middleware()
        self.model = model_factory.construct(config.model)
        self.sessions: Dict[str, ScholarAgentSession] = {}
        self.session_info: Dict[str, Dict] = {}  # Store metadata about sessions

    def setup_middleware(self):
        """Set up CORS middleware if enabled"""
        if self.config.server.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    def setup_routes(self):
        """Set up API routes"""
        self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)(
            self.chat_completions
        )
        self.app.get("/v1/sessions", response_model=SessionListResponse)(
            self.list_sessions
        )
        self.app.get("/health")(self.health_check)

    async def health_check(self):
        """Health check endpoint"""
        return {"status": "ok"}

    async def list_sessions(self) -> SessionListResponse:
        """List all active sessions"""
        sessions = [
            SessionInfo(
                id=session_id,
                created=info["created"],
                last_used=info["last_used"],
            )
            for session_id, info in self.session_info.items()
        ]
        return SessionListResponse(sessions=sessions)

    async def get_or_create_session(
        self, session_id: Optional[str] = None
    ) -> Tuple[str, ScholarAgentSession]:
        """Get an existing session or create a new one"""
        current_time = int(asyncio.get_event_loop().time())

        if session_id and session_id in self.sessions:
            # Update last used time
            self.session_info[session_id]["last_used"] = current_time
            return session_id, self.sessions[session_id]

        # Create a new session
        session = ScholarAgentSession(
            model=self.model,
            config=self.config,
        )
        await session.start()

        # Use the session's UUID as the session ID if none was provided
        new_session_id = session_id or session.uuid

        # Store the session and its metadata
        self.sessions[new_session_id] = session
        self.session_info[new_session_id] = {
            "created": current_time,
            "last_used": current_time,
        }

        return new_session_id, session

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        req: Request,
        token: Optional[HTTPBearer] = Depends(security),
    ) -> ChatCompletionResponse:
        """
        OpenAI-compatible chat completions endpoint
        """
        try:
            # Check authentication if enabled
            if self.config.server.enable_auth and token is None:
                raise HTTPException(status_code=401, detail="Authentication required")
            # Get session ID from header or create a new session
            session_id = req.headers.get(self.config.server.session_header)

            session_id, session = await self.get_or_create_session(session_id)

            # Process the messages
            user_message = ""
            for msg in request.messages:
                if msg.role == "user":
                    user_message = msg.content
                    break

            if not user_message:
                raise HTTPException(
                    status_code=400, detail="No user message found in request"
                )

            # Process the user input
            # DEBUG -- Return the user response!
            await session.user_input(user_message)

            # Get the last message from the supervisor
            supervisor_messages = session.state["agent_messages"]["supervisor"]
            if supervisor_messages and len(supervisor_messages) > 0:
                response_content = supervisor_messages[-1].content
            else:
                response_content = "No response generated"

            # Create the response
            return ChatCompletionResponse(
                id=f"chatcmpl-{session.uuid}",
                created=int(asyncio.get_event_loop().time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant", content=response_content
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=100,  # Placeholder values
                    completion_tokens=100,
                    total_tokens=200,
                ),
            )
        except Exception as e:
            log.error(f"Error processing chat completion: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error processing request: {str(e)}"
            )

    async def cleanup_sessions(self):
        """Clean up inactive sessions"""
        # This could be called periodically to clean up old sessions
        # For now, we'll just implement the method signature
        pass

    def run(self):
        """Run the server"""
        server_config = self.config.server

        # Set up SSL context if TLS is configured
        ssl_context = None
        if server_config.tls_cert_path and server_config.tls_key_path:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                certfile=server_config.tls_cert_path, keyfile=server_config.tls_key_path
            )

            # Configure mTLS if client CA cert is provided
            if server_config.client_ca_cert_path:
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                ssl_context.load_verify_locations(
                    cafile=server_config.client_ca_cert_path
                )
                log.info("mTLS enabled with client certificate verification")
            else:
                log.info("TLS enabled without client certificate verification")

        # Start the server
        log.info(f"Starting server on {server_config.host}:{server_config.port}")
        uvicorn.run(
            self.app,
            host=server_config.host,
            port=server_config.port,
            ssl_keyfile=server_config.tls_key_path,
            ssl_certfile=server_config.tls_cert_path,
            ssl_ca_certs=server_config.client_ca_cert_path,
        )


def create_server(config_path: Optional[str] = None) -> ScholarServer:
    """Create a new ScholarServer instance"""
    if config_path:
        loaded_config = aconfig.Config.from_yaml(config_path)
    else:
        # Use default config path
        loaded_config = aconfig.Config.from_yaml("src/scholar_agent/config/config.yaml")

    return ScholarServer(loaded_config)


def main():
    """Main entry point for the server"""
    # Standard
    import argparse

    parser = argparse.ArgumentParser(description="Scholar Agent Server")

    # Model arguments (mirroring scholar_langgraph.py)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=config.model.config.model,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        "-mp",
        type=str,
        default=config.model.type,
        help="The provider to use for the model",
    )
    parser.add_argument(
        "--ollama-host",
        "-oh",
        type=str,
        default=None,
        help="Non-default host to use for ollama",
    )

    # Server configuration arguments
    parser.add_argument(
        "--port",
        type=int,
        default=config.server.port,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.server.host,
        help="Host to run the server on",
    )
    parser.add_argument(
        "--tls-cert-path",
        type=str,
        default=config.server.tls_cert_path,
        help="Path to TLS certificate file",
    )
    parser.add_argument(
        "--tls-key-path",
        type=str,
        default=config.server.tls_key_path,
        help="Path to TLS key file",
    )
    parser.add_argument(
        "--client-ca-cert-path",
        type=str,
        default=config.server.client_ca_cert_path,
        help="Path to client CA certificate file for mTLS",
    )
    parser.add_argument(
        "--session-header",
        type=str,
        default=config.server.session_header,
        help="HTTP header name for session ID",
    )
    parser.add_argument(
        "--enable-auth",
        action="store_true",
        default=config.server.enable_auth,
        help="Enable bearer token authentication",
    )
    parser.add_argument(
        "--enable-cors",
        action="store_true",
        default=config.server.enable_cors,
        help="Enable CORS middleware",
    )

    # Logging arguments (mirroring scholar_langgraph.py)
    parser.add_argument(
        "--log-level",
        "-l",
        default=config.log.level,
        help="Default logging level",
    )
    parser.add_argument(
        "--log-filters",
        "-lf",
        default=config.log.filters,
        help="Per-channel log filters",
    )
    parser.add_argument(
        "--log-json",
        "-lj",
        action="store_true",
        default=config.log.json,
        help="Use json log formatter",
    )
    parser.add_argument(
        "--log-thread-id",
        "-lt",
        action="store_true",
        default=config.log.thread_id,
        help="Log the thread ID with each log message",
    )

    args = parser.parse_args()

    # Configure logging
    alog.configure(
        default_level=args.log_level,
        filters=args.log_filters,
        formatter="json" if args.log_json else "pretty",
        thread_id=args.log_thread_id,
    )

    # Update config with command-line arguments
    config._config.model.type = args.model_provider
    config._config.model.config.model = args.model
    if args.ollama_host:
        config._config.model.config.base_url = args.ollama_host

    # Update server config
    config._config.server.port = args.port
    config._config.server.host = args.host
    config._config.server.tls_cert_path = args.tls_cert_path
    config._config.server.tls_key_path = args.tls_key_path
    config._config.server.client_ca_cert_path = args.client_ca_cert_path
    config._config.server.session_header = args.session_header
    config._config.server.enable_auth = args.enable_auth
    config._config.server.enable_cors = args.enable_cors

    # Create and run the server
    server = ScholarServer(config._config)
    server.run()


if __name__ == "__main__":
    main()
