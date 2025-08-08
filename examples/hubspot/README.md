# 📇 HubSpot CRM Agent

This example showcases an LLM-powered agent that connects to the **HubSpot CRM API** to streamline customer support workflows such as creating tickets and scheduling meetings — all using natural language.

Built for developers who want to automate CRM actions with minimal setup and maximum flexibility.

---

## 🚀 Quickstart

Run the agent:
```bash
python run.py \
  --input-dir ./examples/hubspot \
  --llm_provider openai \
  --model gpt-4o-mini
````

> 🔑 **Note**: Make sure to set your `access_token` in [`taskgraph.json`](./taskgraph.json) before testing live requests.

---

## ⚙️ Key Features

* ⚡ **Instant CRM Actions** – Create tickets, book meetings, and more without leaving the chat.
* 🔌 **HubSpot API Integration** – Built-in tools wrap official HubSpot APIs with secure token support.
* 🧠 **Natural Language Execution** – Automatically maps user prompts to tool calls using function-calling.

---

## 🛠️ Tool Examples

### 📝 `create_ticket`

Creates a support ticket linked to a customer’s contact ID.

```python
create_ticket(
  cus_cid: str,
  issue: str,
  cus_fname: str,
  cus_lname: str
)
```

**Prompt**:

> "Log a ticket for Jane Doe — she's having trouble logging into her dashboard."

**Response**:

* Returns the new ticket ID after successful creation and association with Jane's contact record.

---

### 📅 `create_meeting`

Schedules a customer meeting with a representative using a link slug and time preferences.

```python
create_meeting(
  cus_fname: str,
  cus_lname: str,
  cus_email: str,
  meeting_date: str,
  meeting_start_time: str,
  duration: int,
  slug: str,
  time_zone: str
)
```

**Prompt**:

> "Book a 30-minute call with John Smith ([john@smith.com](mailto:john@smith.com)) on Friday at 10am PST."

**Response**:

* Returns a confirmation with the scheduled time and meeting link.

