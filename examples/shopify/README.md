# 🛍️ Shopify Integration Agent

A plug-and-play, function-calling agent that connects directly to your **Shopify store** via the Admin API to handle product search, detailed catalog lookups, and customer data retrieval — all in real time.

---

## 🚀 Quickstart

Run the agent:
```bash
python run.py \
  --input-dir ./examples/shopify \
  --llm_provider openai \
  --model gpt-4o-mini
````

> 🧪 **Note**: Make sure to update `admin_token` and `shop_url` in [`taskgraph.json`](./taskgraph.json) before running against live data.

---

## ⚙️ Core Features

* 🔁 **Tool-Driven Execution** – Uses function-calling to dynamically invoke the right API tools based on user prompts.
* 🔐 **Secure & Configurable** – Tools are preconfigured with Shopify Admin API access for easy integration.
* 🧠 **Real-Time Responses** – No need to pre-load products or customers; the agent pulls live data when queried.

---

## 🛠️ Tool Examples

### 🔍 `search_products`

Find products using natural language queries.

```python
search_products(product_query: str)
```

**Prompt**:

> "Do you sell green hats?"

**Response**:

* Answer with conversational context
* List of matching products: titles, links, images, variants

---

### 📦 `get_web_product`

Fetch full product details by product ID.

```python
get_web_product(web_product_id: str)
```

**Prompt**:

> "Tell me more about product 9898304930113"

**Response**:

* Product title, description, pricing
* Inventory levels, variant breakdown, product URL