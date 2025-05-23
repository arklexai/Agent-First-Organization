# DatabaseWorker

## Introduction

Database Workers can handle customer inquiries and take correct operations. Different from other workers, this worker can be a highly customized one capable of solving different tasks. For each task, there will be a different database to interact with.

In this tutorial, we will focus on a specific use case: a show booking system. For such a customer service, it always involves operations like search, book a show, update booking information, and cancel a booking.

## Database Construction

### Database Schema

For a show booking database, a simple design is to include three tables in the database: `show`, `user`, and `booking`. For the `show` table, it includes information of all available shows; the `user` table contains information of registered users in the booking system; the `booking` table has all booking records. A

- Database Schema: show

| Column Name       | Data Type      | Constraints         |
|-------------------|----------------|---------------------|
| `id`              | `VARCHAR(40)` | Primary Key         |
| `show_name`       | `VARCHAR(100)`|                     |
| `genre`           | `VARCHAR(40)` |                     |
| `date`            | `DATE`        |                     |
| `time`            | `TIME`        |                     |
| `description`     | `TEXT`        |                     |
| `location`        | `VARCHAR(100)`|                     |
| `price`           | `DECIMAL`     |                     |
| `available_seats` | `INTEGER`     |                     |

- Database Schema: user

| Column Name   | Data Type      | Constraints         |
|---------------|----------------|---------------------|
| `id`          | `VARCHAR(40)` | Primary Key         |
| `first_name`  | `VARCHAR(40)` |                     |
| `last_name`   | `VARCHAR(40)` |                     |
| `email`       | `VARCHAR(60)` |                     |
| `register_at` | `TIMESTAMP`   |                     |
| `last_login`  | `TIMESTAMP`   |                     |

- Database Schema: booking

| Column Name   | Data Type      | Constraints                 |
|---------------|----------------|-----------------------------|
| `id`          | `VARCHAR(40)` | Primary Key                  |
| `show_id`     | `VARCHAR(40)` | Foreign Key (`show(id)`)     |
| `user_id`     | `VARCHAR(40)` | Foreign Key (`user(id)`)     |
| `created_at`  | `TIMESTAMP`   |                              |

### Populate the database

You can construct the database using `SQLite` and populate it with some sample shows and users. For subsequent tests, please create at least one show and one user, and note down the user id.

## Slot Filling

After each round of conversation, slots are extracted and matched against their original values in the database. For instance, if the user says, “I would like to watch Carmen on Tuesday,” the extracted slots for the show name and time will be mapped to “Carmen” and “2024-11-15,” respectively. Those verified slots will be used in future tasks such as searching, booking, and etc.

## Action Verification

As defined in [TaskGraph](../Taskgraph/Generation.md), there is an attribute `task` for each node, which will be used to determine the corresponding database action. In the sample database worker, we define several actions:

- SearchShow: Search for shows
- BookShow: Book a show
- CheckBooking: Check details of booked show(s)
- CancelBooking: Cancel a booking

## Sample Conversation

After introducing each individual component of the Database Worker, we will try searching and booking functions for a sample conversation:

```json title="dbworker_function_sample.json"
Bot: Hello! How can I help you today?
You: Do you have recommendation for shows?
Bot: Could you let me know your preferences or any specific details you're looking for in an event, like the type of show, location, or date?
You: I want to watch the magic flute.
Bot: Based on your interest in "The Magic Flute," here are some available shows for you to consider. They are all taking place at the Lyric Opera of Chicago in Chicago, IL. The performances are scheduled for November 15th, 20th, and 25th, 2024, each starting at 6:30 PM. This whimsical opera by Wolfgang Amadeus Mozart is priced at $130 per ticket. Please let me know if any of these options appeal to you.
You: I want to watch it on Nov 25.
Bot: Are you interested in exploring more options, or would you like help with booking the show on November 25th?
You: Book the show.
Bot: The show has been booked. Thank you for using our service today! We would love to hear your feedback on how we can improve. Do you have any comments or suggestions?
```

You can checkout the [LangSmith Trace](https://smith.langchain.com/public/f061143d-b4f8-42d6-9bf6-fa678d956d4e/r) when searching available shows based on the user's query. The TaskGraph first calls the Database Worker to verify that the user would like to use the search function, then search for relevant shows based on extracted slots, and finally return searched results.
