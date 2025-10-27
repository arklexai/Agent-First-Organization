import json
from typing import Any
from unittest.mock import Mock, patch

import requests

from integration_tests.utils.base import BaseTestOrchestrator, ChatRole


def create_mock_response(url: str, method: str, **kwargs: dict[str, Any]) -> Mock:
    mock_response = Mock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    request_body = kwargs.get("json", {})

    if url == "https://api.test.com/order" and method == "POST":
        order = request_body.get("order", {})
        if (
            order["supplier"] == "alsfk4kslglskekdo238g5"
            and isinstance(order["product"], str)
            and isinstance(order["quantity"], int)
        ):
            json_data = {
                "status": "success",
                "message": "Order placed successfully",
                "order_id": "ORD-12345",
                "order": order,
            }
        else:
            raise ValueError(f"Invalid order: {order}")

    elif url == "https://api.test.com/search" and method == "GET":
        query = request_body.get("query", "")
        category = request_body.get("category", "")
        if category == "electronics" and isinstance(query, str):
            json_data = {
                "status": "success",
                "message": "Search results found",
                "results": [
                    {
                        "product": "Laptop",
                        "price": 999.99,
                        "category": "electronics",
                    },
                ],
                "total": 2,
            }
        else:
            raise ValueError(f"Invalid search request: {request_body}")

    elif url == "https://api.test.com/items" and method == "POST":
        items = request_body.get("items", [])
        if (
            isinstance(items, list)
            and all(isinstance(item, dict) for item in items)
            and all(
                "productName" in item and "quantity" in item and "unitPrice" in item
                for item in items
            )
            and all(
                isinstance(item["productName"], str)
                and isinstance(item["quantity"], int)
                and isinstance(item["unitPrice"], float)
                for item in items
            )
        ):
            json_data = {
                "status": "success",
                "message": "Items created successfully",
                "items_count": len(items),
                "items": items,
            }
        else:
            raise ValueError(f"Invalid items request: {request_body}")

    elif url == "https://api.test.com/document" and method == "POST":
        title = request_body.get("title", "")
        tags = request_body.get("tags", [])
        if (
            isinstance(title, str)
            and isinstance(tags, list)
            and all(isinstance(tag, str) for tag in tags)
        ):
            json_data = {
                "status": "success",
                "message": "Document created successfully",
                "document_id": "DOC-12345",
                "title": title,
                "tags": tags,
            }
        else:
            raise ValueError(f"Invalid document request: {request_body}")

    else:
        raise ValueError(
            f"Invalid request: {request_body} for endpoint: {url} with method: {method}"
        )

    mock_response.json.return_value = json_data
    mock_response.text = json.dumps(json_data)
    return mock_response


@patch("arklex.resources.tools.custom_tools.http_tool.requests.request")
async def test_slot_filling_agent_comprehensive(mock_request: Mock) -> None:
    orchestrator = BaseTestOrchestrator(
        "integration_tests/taskgraphs/slot_filling_agent_taskgraph.json"
    )
    params = BaseTestOrchestrator.init_params()
    chat_history, params = params["chat_history"], params["parameters"]

    # Agent start message
    text = "<start>"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]

    mock_request.side_effect = create_mock_response

    # Test 1: searchProducts with fixed category value
    text = "Search for gaming laptops"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    ## Verify the HTTP request was made
    search_calls = [
        call
        for call in mock_request.call_args_list
        if call[1].get("url") == "https://api.test.com/search"
        and call[1].get("method") == "GET"
    ]
    assert len(search_calls) >= 1, "searchProducts should have been called"
    search_call = search_calls[0]
    assert search_call[1]["url"] == "https://api.test.com/search"
    assert search_call[1]["method"] == "GET"
    assert search_call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert search_call[1]["headers"]["Content-Type"] == "application/json"
    ## Validate request body
    request_body = search_call[1]["json"]
    assert request_body["category"] == "electronics", (
        "Fixed category value must match the taskgraph configuration"
    )
    assert isinstance(request_body["query"], str), "Query must be a string"
    assert isinstance(request_body["category"], str), "Category must be a string"
    assert "laptop" in output["answer"].lower() or "product" in output["answer"].lower()

    # Test 2: orderProduct with fixed supplier value
    mock_request.reset_mock()
    mock_request.side_effect = create_mock_response
    text = "I want to order 5 laptops"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    ## Verify the HTTP request was made
    order_calls = [
        call
        for call in mock_request.call_args_list
        if call[1].get("url") == "https://api.test.com/order"
        and call[1].get("method") == "POST"
    ]
    assert len(order_calls) >= 1, "orderProduct should have been called"
    order_call = order_calls[0]
    assert order_call[1]["url"] == "https://api.test.com/order"
    assert order_call[1]["method"] == "POST"
    assert order_call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert order_call[1]["headers"]["Content-Type"] == "application/json"
    ## Validate request body
    request_body = order_call[1]["json"]
    assert "order" in request_body
    order = request_body["order"]
    assert order["supplier"] == "alsfk4kslglskekdo238g5", (
        "Fixed supplier value must match the taskgraph configuration"
    )
    assert isinstance(order["product"], str), "Product must be a string"
    assert isinstance(order["quantity"], int), "Quantity must be an integer"
    assert isinstance(order["supplier"], str), "Supplier must be a string"
    ## Verify successful order response
    assert "success" in output["answer"].lower() or "order" in output["answer"].lower()

    # Test 3: createProductItems with array validation
    mock_request.reset_mock()
    mock_request.side_effect = create_mock_response
    text = "Create items: laptop quantity 2 price 999.99, mouse quantity 5 price 29.99"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    ## Verify the HTTP request was made
    items_calls = [
        call
        for call in mock_request.call_args_list
        if call[1].get("url") == "https://api.test.com/items"
        and call[1].get("method") == "POST"
    ]
    assert len(items_calls) >= 1, "createProductItems should have been called"
    items_call = items_calls[0]
    assert items_call[1]["url"] == "https://api.test.com/items"
    assert items_call[1]["method"] == "POST"
    ## Validate request body
    request_body = items_call[1]["json"]
    assert "items" in request_body
    items = request_body["items"]
    assert isinstance(items, list), "Items must be an array"
    assert len(items) >= 1, "Items array should contain at least one item"
    ## Validate each item in the array
    for item in items:
        ## Validate each item
        assert isinstance(item, dict), "Each item must be an object"
        assert "productName" in item, "Each item must have productName"
        assert "quantity" in item, "Each item must have quantity"
        assert "unitPrice" in item, "Each item must have unitPrice"

        assert isinstance(item["productName"], str), "productName must be a string"
        assert isinstance(item["quantity"], int), "quantity must be an integer"
        assert isinstance(item["unitPrice"], float), "unitPrice must be a float"
    ## Verify successful creation response
    assert (
        "success" in output["answer"].lower() or "created" in output["answer"].lower()
    )

    # Test 4: createDocument with array of strings validation
    mock_request.reset_mock()
    mock_request.side_effect = create_mock_response
    text = "Create a document titled 'Product Catalog' with tags: products, catalog, inventory, electronics"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    ## Verify the HTTP request was made
    document_calls = [
        call
        for call in mock_request.call_args_list
        if call[1].get("url") == "https://api.test.com/document"
        and call[1].get("method") == "POST"
    ]
    assert len(document_calls) >= 1, "createDocument should have been called"
    document_call = document_calls[0]
    assert document_call[1]["url"] == "https://api.test.com/document"
    assert document_call[1]["method"] == "POST"
    assert document_call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert document_call[1]["headers"]["Content-Type"] == "application/json"
    ## Validate request body
    request_body = document_call[1]["json"]
    assert "title" in request_body
    assert "tags" in request_body
    assert isinstance(request_body["title"], str), "Title must be a string"
    assert isinstance(request_body["tags"], list), "Tags must be an array"
    assert len(request_body["tags"]) >= 1, "Tags array should contain at least one tag"
    for tag in request_body["tags"]:
        assert isinstance(tag, str), "Each tag must be a string (base type)"
    ## Verify successful document creation response
    assert (
        "success" in output["answer"].lower() or "document" in output["answer"].lower()
    )
