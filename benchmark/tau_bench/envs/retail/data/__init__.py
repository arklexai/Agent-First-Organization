# Copyright Sierra

import json
import os
from typing import Any, Final, Dict

FOLDER_PATH: Final[str] = os.path.dirname(__file__)


def load_data() -> Dict[str, Any]:
    with open(os.path.join(FOLDER_PATH, "orders.json")) as f:
        order_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "products.json")) as f:
        product_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "users.json")) as f:
        user_data = json.load(f)
    return {
        "orders": order_data,
        "products": product_data,
        "users": user_data,
    }
