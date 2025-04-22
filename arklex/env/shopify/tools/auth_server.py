"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: auth_server

This file contains the code for a Flask server that listens for an auth token from a Shopify app.
"""
import ngrok
import threading
from flask import Flask, request
import time

from dotenv import load_dotenv
load_dotenv()

auth_token = None

def start_auth_server():
    # Start a temporary Flask server to listen for auth token
    app = Flask(__name__)

    @app.route('/callback', methods=['GET'])
    def callback():
        global auth_token
        # Get token from query parameters
        token = request.args.get('code')
        state = request.args.get('state')
        print("Received auth token:", token)
        
        auth_token = token
        
        # Save the token to a log file
        with open("auth_log.txt", "a") as log_file:
            log_file.write(f"Received at {time.ctime()}: {token}\n")

        # Shut down the Flask server
        shutdown_func = request.environ.get('werkzeug.server.shutdown')
        if shutdown_func:
            shutdown_func()

        return  """
        <script>
            window.close();
        </script>
        """
        # return {"status": "received"}, 200

    # Run the Flask server in the main thread (blocking call)
    app.run(port=8000)

def authenticate_server():
    global auth_token

    # Start Ngrok tunnel to localhost:8000
    ngrok_tunnel = ngrok.forward(8000, authtoken_from_env=True, domain="causal-bluejay-humble.ngrok-free.app")
    print(f"Waiting to authenticate...")

    # Run Flask server in a separate thread
    server_thread = threading.Thread(target=start_auth_server)
    server_thread.start()

    # Wait until auth token is received
    while auth_token is None:
        time.sleep(1)

    # Close Ngrok tunnel
    ngrok.disconnect(ngrok_tunnel.url())

    # print(auth_token)
    
    print(f"Authenticated!")
    return auth_token

if __name__ == '__main__':
    authenticate_server()