# Customer Service Agent (RAG)

This example demonstrates a **Retrieval-Augmented Generation (RAG)** agent built to handle customer support inquiries for **[Richtech Robotics](https://richtechrobotics.com/)** — a company specializing in commercial service robots for cleaning, delivery, and production.


## 🚀 Quickstart

### Run the agent
```bash
python run.py \
  --input-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o-mini
```

## 💬 Example Utterance

**User**:  
> *What cleaning robots do you sell?*

**Agent (via RAG)**:  
> *We offer two commercial cleaning robots: the DUST-E SX and the DUST-E MX. These robots are designed for business purposes to help with various cleaning tasks. Delivery for a commercial cleaning robot typically takes about two months. Can you tell me about your specific needs or requirements for robotics solutions?*

---

## 📄 Built-in Web Scraping for RAG

This agent automatically scrapes and indexes content from Richtech Robotics' website for use in RAG-based answers. You don't need to upload PDFs or write custom crawlers — just specify the URLs in the config.

### ✅ Defined in `customer_service_config.json`:
```json
"rag_docs": [
  {
    "source": "https://www.richtechrobotics.com/",
    "type": "url",
    "num": 20
  }
]
```
- **source**: The target website to scrape

- **type**: Set to "url" to enable web-based scraping

- **num**: Limits how many pages are fetched