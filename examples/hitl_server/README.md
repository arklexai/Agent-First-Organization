# 🧑‍💼 Human-in-the-Loop (HITL) Agent

This example demonstrates how to inject **live human feedback or decision points** into an LLM workflow. The HITL agent supports both chat and multiple-choice interactions using stateful workers.

Ideal for production scenarios where automated responses need to defer to a human — e.g., escalations, approvals, or confirmations.

---

## 🚀 Quickstart

Run the agent:
```bash
python run.py \
  --input-dir ./examples/hitl_server \
  --llm_provider openai \
  --model gpt-4o-mini
````

---

## 🧠 Key Components

* ✅ **Flag-Based State Management** – Workers track interaction state using metadata flags.
* 🧵 **Chat and Multiple Choice Modes** – Choose between free-form messaging and structured decision points.

---

## 🔧 Worker Types

### 💬 `HITLWorkerChatFlag`

Production-ready worker that handles live chat escalation.

```python
class HITLWorkerChatFlag(HITLWorker):
    mode = "chat"
```

**Behavior**:

* Detects when a human should be looped in
* Returns `"I'll connect you to a representative!"`
* Manages conversation lifecycle via flags (`hitl: "live"`)

---

### ✅ `HITLWorkerMCFlag`

Handles yes/no multiple-choice decisions (e.g. purchase confirmation).

```python
class HITLWorkerMCFlag(HITLWorker):
    mode = "mc"
```

**Behavior**:

* Prompts: `"Should the user continue with this purchase? (Y/N)"`
* Tracks retries, fallback behavior
* Returns a result like `"User is allowed to continue with the purchase"`

---

## 🧪 Testing Locally

To experiment with chat or MC flows in isolation, use:

* `HITLWorkerTestChat` – mock chat worker for local testing
* `HITLWorkerTestMC` – mock MC interaction without real-time feedback

---