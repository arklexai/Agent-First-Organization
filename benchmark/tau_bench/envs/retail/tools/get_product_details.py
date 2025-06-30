import json
from typing import Any

from benchmark.tau_bench.envs.tool import Tool


class GetProductDetails(Tool):
    @staticmethod
    def invoke(data: dict[str, Any], product_id: str) -> str:
        products = data["products"]
        if product_id in products:
            return json.dumps(products[product_id])
        raise Exception("Error: product not found")

    @staticmethod
    def get_info() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_product_details",
                "description": "Get the inventory details of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "The product id, such as '6086499569'. Be careful the product id is different from the item id.",
                        },
                    },
                    "required": ["product_id"],
                },
            },
        }
