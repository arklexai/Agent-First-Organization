# Chat Models

## Supported Language Models

### Providers and Models

| Provider | Models | Tool Calling Support |
|----------|--------|---------------------|
| **OpenAI** (Default) | `gpt-4o`, `gpt-4o-mini`, `gpt-4.5-preview` | ✅ Full support |
| **Google Gemini** | `gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-2.0-flash-lite` | ✅ Only `gemini-2.0-flash` |
| **Anthropic** | `claude-3-5-haiku-20241022`, `claude-3-haiku-20240307`, `claude-3-7-sonnet-20250219` | ✅ Full support |
| **Hugging Face** | `microsoft/Phi-3-mini-4k-instruct` | ❌ Not supported |

## Configuration

### API Keys Setup

Add the appropriate API key to your `.env` file:

- **OpenAI**: `OPEN_API_KEY`
- **Google Gemini**: `GOOGLE_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Hugging Face**: `HUGGINGFACE_API_KEY` ([Get token here](https://huggingface.co/docs/api-inference/en/getting-started#getting-a-token))

## Usage

### Taskgraph Generation

Generate taskgraphs using the `create.py` script:

```bash
python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service --model <MODEL> --llm_provider <PROVIDER>
```

**Examples:**

- OpenAI: `--model gpt-4o-mini --llm_provider openai`
- Google Gemini: `--model gemini-2.0-flash --llm_provider gemini`
- Anthropic: `--model claude-3-5-haiku-20241022 --llm_provider anthropic`

**Note:** Hugging Face is not supported for taskgraph generation (will be implemented in the future).

### Running the Bot

Run the bot using the `run.py` script:

```bash
python run.py --input-dir ./examples/customer_service --model <MODEL> --llm_provider <PROVIDER>
```

**Examples:**

- OpenAI: `--model gpt-4o-mini --llm_provider openai`
- Google Gemini: `--model gemini-2.0-flash-lite --llm_provider gemini`
- Anthropic: `--model claude-3-5-haiku-20241022 --llm_provider anthropic`
- Hugging Face: `--model microsoft/Phi-3-mini-4k-instruct --llm_provider huggingface`

**Note:** For Hugging Face models, you may need to request access to specific models via the Hugging Face website.

## Evaluation

For evaluation and testing, you can also use the model API server:

1. First, start the model API server (defaults to OpenAI with "gpt-4o-mini" model):

```bash
python model_api.py --input-dir ./examples/customer_service
```

2. Then run evaluation (in a separate terminal):

```bash
python eval.py --model_api http://127.0.0.1:8000/eval/chat --config "examples/customer_service/customer_service_config.json" --documents_dir "examples/customer_service" --model "claude-3-haiku-20240307" --llm_provider "anthropic" --task "all"
```

## Sample Conversation

Run the bot with:

```bash
python run.py --input-dir ./examples/customer_service --model claude-3-5-haiku-20241022 --llm_provider anthropic
```

### Example Output

> **ASSISTANT**: Hello! I'm your Customer Service Assistant. How can I assist you with your inquiries today? Whether you need information about our products, services, or policies, or require help resolving an issue, I'm here to help!
>
> **USER**: What products do you offer?
>
> **ASSISTANT**: At Richtech Robotics, we offer a range of advanced business-oriented robots across several categories:
>
> **Worker Robots:**
>
> - ADAM (multipurpose robot bartender for making tea, coffee, and cocktails)
> - ARM
> - ACE
>
> **Delivery Robots:**
>
> - Matradee
> - Matradee X
> - Matradee L
> - Richie
>
> **Cleaning Robots:**
>
> - DUST-E SX
> - DUST-E MX
>
> **Multipurpose Robots:**
>
> - Skylark
>
> Our robots are designed for business applications, not home use. The ADAM robot is particularly unique, as it's available for both purchase and rental, and we even operate the world's first robot milk tea shop, ClouTea, in Las Vegas using this robot.
>
> Can you please provide more details about the robot you are interested in, such as the model name or specific features you are curious about?
>
> **USER**: Tell me more about the worker robots, what are their differences?
>
> **ASSISTANT**: Let me provide you with details about our worker robots:
>
> **ADAM Robot:**
>
> - Multipurpose bartending robot
> - Can prepare tea, coffee, and cocktails
> - Available for purchase and rental
> - Featured in our ClouTea milk tea shop in Las Vegas
> - Typical delivery time: 2 weeks
>
> **ARM Robot:**
>
> - Specialized worker robot
> - Specific details would require further consultation
> - Used in various industrial and business settings
>
> **ACE Robot:**
>
> - Another worker robot in our lineup
> - Designed for specific industrial applications
> - Detailed specifications would need a more in-depth discussion
>
> Each of these robots has unique capabilities tailored to different business needs. Would you like me to elaborate on any specific features or help you determine which robot might best suit your requirements?
>
> **USER**: I'm interested in ADAM, can we book a demo?
>
> **ASSISTANT**: Absolutely! I can help you book a demo for our ADAM robot. We offer several ways to schedule a demonstration:
>
> 1. **Online Booking**: Visit our robot rentals page at [https://www.richtechrobotics.com/robot-rentals#rent-new-form](https://www.richtechrobotics.com/robot-rentals#rent-new-form) to schedule a demo.
>
> 2. **Contact Our Sales Team**: Reach our sales line at (866) 236-3835 to discuss a demo and explore how ADAM can benefit your business.
>
> To proceed with booking a demo, I'll need a bit more information from you:
>
> - What is your name?
> - What is your company name?
> - What is the best contact number and email address to reach you?
> - Are you interested in a demo for an event, business use, or general exploration?
>
> Once I have these details, I can help guide you through the demo scheduling process and connect you with our sales team who can provide more specific information about the ADAM robot's capabilities.
