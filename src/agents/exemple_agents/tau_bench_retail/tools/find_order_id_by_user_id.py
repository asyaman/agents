# Copyright Sierra

import json
from typing import Any, Dict

from agents.exemple_agents.tau_bench_retail.tools.tool import Tool


class FindOrderIdsByUserId(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], user_id: str) -> str | str:
        user_orders = data["users"][user_id]["orders"]
        if user_orders:
            return json.dumps(user_orders)
        else:
            return "Error: user id not found"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "find_order_ids_by_user_id",
                "description": "find all orders by user id. If the user id is not identified, the function will return an error message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user id, such as 'sara_doe_496'.",
                        },
                    },
                    "required": ["user_id"],
                },
            },
        }
