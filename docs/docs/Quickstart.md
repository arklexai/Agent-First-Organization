# Quickstart
This guide walks you through getting started with Arklex—from running your first agent to building more advanced workflows.

## 1. Project Setup
### Clone repo
```bash
git clone https://github.com/arklexai/Agent-First-Organization.git
```

### Create the virtual environment
```bash
cd Agent-First-Organization
python3 -m venv .venv
```
### Activate the virtual environment
```bash
source .venv/bin/activate 
```
### Install Arklex
```bash
pip install arklex
```

### Set Your OpenAI API Key
If you don't have one, [get your API key here](https://platform.openai.com/account/api-keys).
```bash
export OPENAI_API_KEY=sk-...
```


## 2. Run Customer Service Agent
Arklex comes with an example to get you started fast:
```bash
python run.py --input-dir ./examples/customer_service
```
You should see a conversational flow like this:
```text
Bot: Hello! I'm here to assist you with any customer service inquiries you may have.
Whether you need information about our products, services, or policies, or if you need
help with any issues or transactions, feel free to ask. How can I assist you today?

You: what products do you sell

Bot: We offer a variety of robotics products designed primarily for business purposes.
Our lineup includes:

- **Worker Robots:** ADAM, ARM, ACE  
- **Delivery Robots:** Matradee, Matradee X, Matradee L, Richie  
- **Cleaning Robots:** DUST-E SX, DUST-E MX  
- **Multipurpose Robot:** Skylark

The ADAM robot is available for both purchase and rental and is versatile enough for 
various tasks, including making tea, coffee, and cocktails.

If you're looking for specific solutions or have particular needs in mind, I’d love to hear more about them!

```


## 3. Creating Your Own Agent

### Step 1: Copy the Existing Agent Config
Duplicate the example customer service config to start customizing your own agent:
```bash
cp -r examples/customer_service examples/customer_service_copy
```
Now you can edit the config in the new folder:
`examples/customer_service_copy/customer_service_config.json`


### Step 2: Customize the Config
Replace the RAG documents or data files with your own (e.g., company-specific FAQs or website content).

Modify the agent instructions and any tools or workers as needed in the JSON config.

📁 For RAG: update the document paths and content in the `task_docs` and `rag_docs` fields of the config.

Example changes: ...
```jsonc
...
    "domain": "AI agents",
    "intro": "The Arklex AI's core technology is an advanced general-purpose AI Agent platform, which reduces hard-code engineering effort for developers and provides personalized experiences to end users.",
    "task_docs": [
        {
            "source": "https://www.arklex.ai/",
            "type": "url",
            "num": 20
        }
    ],
    "rag_docs": [
        {
            "source": "https://www.arklex.ai/",
            "type": "url",
            "num": 20
        }
    ],...
```
### Step 3: Create Your Agent
Once the config is ready, generate the agent:
```bash
python create.py --config ./examples/customer_service_copy/customer_service_config.json --output-dir ./examples/customer_service_copy
```

### Step 4: Run the Agent
Run your new agent with:
```bash
python run.py --input-dir ./examples/customer_service_copy
```
You should see a conversational flow like this:
```text
"Bot: Hello! I'm here to assist you with any customer service inquiries."
You: what products does Arklex provide
getAPIBotResponse Time: 3.324455976486206
'Bot: Arklex provides a range of products and services tailored to various '
 'needs, primarily focused on an advanced AI Agent platform designed to assist '
 'developers by reducing engineering efforts and enhancing user experiences. '
 "If you're looking for more specifics or details about a particular product, "
 'feel free to ask! Additionally, if you have any questions about company '
 "policies, I'll be happy to help with those as well."
```

### Step 5: Evaluate Your Agent (Optional)
1. Start the model API server (uses OpenAI API with `gpt-4o-mini` by default):
```bash
python model_api.py --input-dir ./examples/customer_service_copy
```
2. In another terminal, run the evaluation:
```bash
python eval.py \
  --model_api http://127.0.0.1:8000/eval/chat \
  --config examples/customer_service_copy/customer_service_config.json \
  --documents_dir examples/customer_service_copy \
  --task all
```


