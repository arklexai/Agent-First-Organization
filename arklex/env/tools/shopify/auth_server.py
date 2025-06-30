"""
This module provides a Flask server implementation for handling Shopify OAuth 2.0 callbacks.
It creates a temporary server that listens for authentication tokens and manages the OAuth flow.

The server:
1. Creates a temporary Flask server to listen for auth tokens
2. Uses ngrok to expose the local server to the internet
3. Handles the OAuth callback and token reception
4. Automatically shuts down after receiving the token

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: auth_server

This file contains the code for a Flask server that listens for an auth token from a Shopify app.
"""

import threading
import time
from collections.abc import Callable

import ngrok
from dotenv import load_dotenv
from flask import Flask, request

load_dotenv()

auth_token: str | None = None


def start_auth_server() -> None:
    """
    Start a temporary Flask server to listen for authentication tokens.

    The server:
    1. Creates a Flask application
    2. Sets up a callback route to handle OAuth responses
    3. Processes the received token and state
    4. Logs the token to a file
    5. Automatically shuts down after receiving the token

    The callback route returns HTML that closes the window after successful authentication.
    """
    app = Flask(__name__)

    @app.route("/callback", methods=["GET"])
    def callback() -> str:
        global auth_token
        # Get token from query parameters
        token = request.args.get("code")
        print("Received auth token:", token)

        auth_token = token

        # Save the token to a log file
        with open("auth_log.txt", "a") as log_file:
            log_file.write(f"Received at {time.ctime()}: {token}\n")

        # Shut down the Flask server
        shutdown_func: Callable | None = request.environ.get("werkzeug.server.shutdown")
        if shutdown_func:
            shutdown_func()

        return """
        <script>
            window.close();
        </script>
        """

    # Run the Flask server in the main thread (blocking call)
    app.run(port=8000)


def authenticate_server() -> str:
    """
    Set up and run the authentication server to receive OAuth tokens.

    This function:
    1. Creates an ngrok tunnel to expose the local server
    2. Starts the Flask server in a separate thread
    3. Waits for the authentication token
    4. Closes the ngrok tunnel after receiving the token

    Returns:
        str: The received authentication token.

    Note:
        This function blocks until a token is received or the server is shut down.
    """
    global auth_token

    # Start Ngrok tunnel to localhost:8000
    ngrok_tunnel = ngrok.forward(
        8000, authtoken_from_env=True, domain="causal-bluejay-humble.ngrok-free.app"
    )
    print("Waiting to authenticate...")

    # Run Flask server in a separate thread
    server_thread = threading.Thread(target=start_auth_server)
    server_thread.start()

    # Wait until auth token is received
    while auth_token is None:
        time.sleep(1)

    # Close Ngrok tunnel
    ngrok.disconnect(ngrok_tunnel.url())

    print("Authenticated!")
    return auth_token


if __name__ == "__main__":
    authenticate_server()
