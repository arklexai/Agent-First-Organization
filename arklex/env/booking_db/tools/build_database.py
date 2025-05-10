import sqlite3
import argparse
from pathlib import Path
import os


def build_database(folder_path):
    db_path = Path(folder_path) / "show_booking_db.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    # Creating the database with a .sqlite extension
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables based on the provided schema
    cursor.execute('''
        CREATE TABLE show (
            id VARCHAR(40) PRIMARY KEY,
            show_name VARCHAR(100),
            genre VARCHAR(40),
            date DATE,
            time TIME,
            description TEXT,
            location VARCHAR(100),
            price DECIMAL,
            available_seats INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE user (
            id VARCHAR(40) PRIMARY KEY,
            first_name VARCHAR(40),
            last_name VARCHAR(40),
            email VARCHAR(60),
            register_at TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE booking (
            id VARCHAR(40) PRIMARY KEY,
            show_id VARCHAR(40),
            user_id VARCHAR(40),
            created_at TIMESTAMP,
            FOREIGN KEY (show_id) REFERENCES show(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
        )
    ''')

    # Populate sample data
    shows = [
        {
            "show_name": "The Dead, 1904",
            "genre": "Opera",
            "date": "2024-11-26",
            "time": "19:30:00",
            "description": "Dot Dot Productions, in association with the American Irish Historical Society, presents Irish Repertory Theatre's production. Based on the novella by James Joyce. Adapted by Paul Muldoon & Jean Hanff Korelitz.",
            "location": "991 Fifth Avenue New York, NY",
            "price": 200.0,
            "available_seats": 200,
            "id": "show_8406f0c6-6644-4a19-9448-670c9941b8d8"
        },
        {
            "show_name": "The Dead, 1904",
            "genre": "Opera",
            "date": "2024-11-30",
            "time": "17:00:00",
            "description": "Dot Dot Productions, in association with the American Irish Historical Society, presents Irish Repertory Theatre's production. Based on the novella by James Joyce. Adapted by Paul Muldoon & Jean Hanff Korelitz.",
            "location": "991 Fifth Avenue New York, NY",
            "price": 300.0,
            "available_seats": 200,
            "id": "show_06d03f1d-c38c-4ab2-b210-3342c76425f5"
        },
        {
            "show_name": "Carmen",
            "genre": "Opera",
            "date": "2024-11-12",
            "time": "20:00:00",
            "description": "A passionate opera by Georges Bizet.",
            "location": "San Francisco Opera, San Francisco, CA",
            "price": 120.0,
            "available_seats": 150,
            "id": "show_c32f2e1f-798a-406d-979b-733c2b37d90c"
        },
        {
            "show_name": "A Child's Christmas in Wales",
            "genre": "Opera",
            "date": "2024-12-04",
            "time": "19:00:00",
            "description": "This December, Irish Rep offers up the seventh special return of its joyous holiday classic, Dylan Thomas's iconic A Child's Christmas in Wales. Charlotte Moore's musical adaptation of this 'never to be forgotten day at the end of the unremembered year' features heartwarming contemporary and traditional Christmas music interwoven with the popular story of that snowy Christmas Day in Wales.",
            "location": "991 Fifth Avenue New York, NY",
            "price": 55.0,
            "available_seats": 180,
            "id": "show_84c92d38-1f01-4251-ac57-334fb8244477"
        },
        {
            "show_name": "A Child's Christmas in Wales",
            "genre": "Opera",
            "date": "2024-12-11",
            "time": "14:00:00",
            "description": "This December, Irish Rep offers up the seventh special return of its joyous holiday classic, Dylan Thomas's iconic A Child's Christmas in Wales. Charlotte Moore's musical adaptation of this 'never to be forgotten day at the end of the unremembered year' features heartwarming contemporary and traditional Christmas music interwoven with the popular story of that snowy Christmas Day in Wales.",
            "location": "Lyric Opera of Chicago, Chicago, IL",
            "price": 45.0,
            "available_seats": 100,
            "id": "show_44df967e-e4ef-44ec-923a-a118be06240d"
        },
        {
            "show_name": "Beckett Briefs",
            "genre": "Opera",
            "date": "2025-01-15",
            "time": "18:30:00",
            "description": "The plays typically explore themes of existentialism, the human condition, and the complexities of life through Beckett's distinctive minimalist and absurdist style.",
            "location": "Houston Grand Opera, Houston, TX",
            "price": 65.0,
            "available_seats": 180,
            "id": "show_c7ac8410-e03d-45e8-9b50-6a9c72b87805"
        },
        {
            "show_name": "The Beacon",
            "genre": "Opera",
            "date": "2024-09-11",
            "time": "19:00:00",
            "description": "It is a psychological drama by Nancy Harris that delves into themes of family estrangement, unresolved trauma, and the elusive nature of truth. The narrative centers on Beiv, a renowned artist who relocates from suburban Dublin to a secluded cottage on an island off the coast of West Cork, Ireland. ",
            "location": "991 Fifth Avenue New York, NY",
            "price": 140.0,
            "available_seats": 160,
            "id": "show_851cd9f3-734d-414e-b75c-5a389dc6a380"
        },
        {
            "show_name": "The Beacon",
            "genre": "Opera",
            "date": "2024-10-17",
            "time": "18:00:00",
            "description": "It is a psychological drama by Nancy Harris that delves into themes of family estrangement, unresolved trauma, and the elusive nature of truth. The narrative centers on Beiv, a renowned artist who relocates from suburban Dublin to a secluded cottage on an island off the coast of West Cork, Ireland. ",
            "location": "991 Fifth Avenue New York, NY",
            "price": 140.0,
            "available_seats": 90,
            "id": "show_0a3babd6-d153-41f7-bab0-8a6a995ffb5a"
        },
        {
            "show_name": "Don Giovanni",
            "genre": "Opera",
            "date": "2024-11-20",
            "time": "19:30:00",
            "description": "A dramatic opera by Wolfgang Amadeus Mozart.",
            "location": "Washington National Opera, Washington, D.C.",
            "price": 155.0,
            "available_seats": 170,
            "id": "show_2be3c426-2822-45dc-84f0-68609ca53f86"
        },
        {
            "show_name": "On Beckett",
            "genre": "Opera",
            "date": "2024-07-10",
            "time": "19:00:00",
            "description": "A dramatic opera by Wolfgang Amadeus Mozart.",
            "location": "Metropolitan Opera House, New York, NY",
            "price": 155.0,
            "available_seats": 170,
            "id": "show_11604b40-6058-4264-8c80-95d774596d12"
        }
    ]

    users = [
        {
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice.smith@gmail.com",
            "register_at": "2024-10-01 09:15:00",
            "last_login": "2024-10-12 08:30:00",
            "id": "user_be6e1836-8fe9-4938-b2d0-48f810648e72"
        },
        {
            "first_name": "Bob",
            "last_name": "Johnson",
            "email": "bob.johnson@gmail.com",
            "register_at": "2024-10-02 10:00:00",
            "last_login": "2024-10-13 07:45:00",
            "id": "user_ffd7218a-31c4-4377-902e-33faf36d168c"
        },
        {
            "first_name": "Carol",
            "last_name": "Williams",
            "email": "carol.williams@gmail.com",
            "register_at": "2024-10-03 11:30:00",
            "last_login": "2024-10-14 09:00:00",
            "id": "user_7404fbd7-d043-4d4c-80e6-28c9ae81dacc"
        },
        {
            "first_name": "David",
            "last_name": "Jones",
            "email": "david.jones@gmail.com",
            "register_at": "2024-10-04 12:00:00",
            "last_login": "2024-10-15 09:30:00",
            "id": "user_42f20628-1989-4d87-81e6-4f4faca63410"
        },
        {
            "first_name": "Eve",
            "last_name": "Brown",
            "email": "eve.brown@gmail.com",
            "register_at": "2024-10-05 13:45:00",
            "last_login": "2024-10-16 10:15:00",
            "id": "user_13074ec4-3813-4bbd-afa4-339e9eee27e9"
        }
    ]

    # Insert data into the database
    for show in shows:
        columns = ', '.join(show.keys())
        placeholders = ', '.join(['?'] * len(show))
        values = tuple(show.values())
        sql = f"INSERT INTO show ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, values)

    for user in users:
        columns = ', '.join(user.keys())
        placeholders = ', '.join(['?'] * len(user))
        values = tuple(user.values())
        sql = f"INSERT INTO user ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, values)

    cursor.execute('''
        INSERT INTO booking (id, show_id, user_id, created_at)
        VALUES
            ('1', 'show_8406f0c6-6644-4a19-9448-670c9941b8d8', 'user_be6e1836-8fe9-4938-b2d0-48f810648e72', '2024-10-12 10:00:00')
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True, type=str, help="location to save the documents")
    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        os.makedirs(args.folder_path)

    build_database(args.folder_path)