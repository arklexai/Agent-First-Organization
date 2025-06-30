"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-20)
    - Intended for future feature expansion

Module Name: chat_server

This file contains the code for setting up a chat server that can accept connections from chat clients and broadcast messages to all connected clients.
"""

import argparse
import asyncio
import json
import sys


class ChatServer:
    """Chat server class"""

    # dict of all current users
    ALL_USERS: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
    SERVER_USER: str = "Server"

    def __init__(self, host_address: str, host_port: int) -> None:
        self.host_address: str = host_address
        self.host_port: int = host_port

    # write a message to a stream writer
    async def write_message(
        self, writer: asyncio.StreamWriter, msg_bytes: bytes
    ) -> None:
        # write message to this user
        writer.write(msg_bytes)
        # wait for the buffer to empty
        await writer.drain()

    # send a message to all connected users
    async def broadcast_message(self, name: str, message: str = "") -> None:
        # report locally

        print(f"{name}: {message.strip()}")
        sys.stdout.flush()
        msg_bytes: bytes = json.dumps({"name": name, "message": message}).encode()
        # enumerate all users and broadcast the message

        # create a task for each write to client
        tasks: list[asyncio.Task] = [
            asyncio.create_task(self.write_message(w, msg_bytes))
            for _, (_, w) in self.ALL_USERS.items()
        ]
        # wait for all writes to complete
        if tasks:
            _ = await asyncio.wait(tasks)

    # connect a user
    async def connect_user(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> str:
        # ask the user for their name
        data: bytes = await reader.read(1024)  # Read raw bytes
        # print(data)
        # convert name to string
        name: str = json.loads(data.decode())["name"]
        # store the user details
        self.ALL_USERS[name] = (reader, writer)
        # announce the user
        await self.broadcast_message(self.SERVER_USER, f"{name} has connected\n")
        # welcome message
        await self.write_message(
            writer,
            json.dumps(
                {
                    "name": self.SERVER_USER,
                    "message": f"Welcome {name}. Send QUIT to disconnect.",
                }
            ).encode(),
        )
        return name

    # disconnect a user
    async def disconnect_user(self, name: str, writer: asyncio.StreamWriter) -> None:
        # close the user's connection
        writer.close()
        await writer.wait_closed()
        # remove from the dict of all users
        del self.ALL_USERS[name]
        # broadcast the user has left
        await self.broadcast_message(self.SERVER_USER, f"{name} has disconnected\n")

    # handle a chat client
    async def handle_chat_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        print("Client connecting...")
        # connect the user
        name: str = await self.connect_user(reader, writer)
        try:
            # read messages from the user
            while True:
                # read data
                data: bytes = await reader.read(1024)  # Read raw bytes
                decoded_data: str = data.decode()
                sys.stdout.flush()
                for data in decoded_data.split("{"):
                    if not data:
                        continue

                    # convert to string
                    data_json: dict[str, str] = json.loads("{" + data)
                    name: str = data_json["name"]
                    line: str = data_json["message"].strip()
                    # check for exit
                    if line == "QUIT":
                        break
                    # broadcast message
                    await self.broadcast_message(name, line)
                else:
                    continue
                break
        finally:
            # disconnect the user
            await self.disconnect_user(name, writer)

    # chat server
    async def main(self) -> None:
        # define the local host
        # create the server
        server: asyncio.Server = await asyncio.start_server(
            self.handle_chat_client, self.host_address, self.host_port
        )
        # run the server
        async with server:
            # report message
            print("Chat Server Running\nWaiting for chat clients...")
            # accept connections
            await server.serve_forever()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--server-address", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8888)
    args: argparse.Namespace = parser.parse_args()

    server: ChatServer = ChatServer(args.server_address, args.server_port)
    # start the event loop
    asyncio.run(server.main())
