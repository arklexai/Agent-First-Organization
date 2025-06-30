import json
from typing import Any

from benchmark.tau_bench.envs.tool import Tool


class ListAllAirports(Tool):
    @staticmethod
    def invoke(data: dict[str, Any]) -> str:
        airports = [
            "SFO",
            "JFK",
            "LAX",
            "ORD",
            "DFW",
            "DEN",
            "SEA",
            "ATL",
            "MIA",
            "BOS",
            "PHX",
            "IAH",
            "LAS",
            "MCO",
            "EWR",
            "CLT",
            "MSP",
            "DTW",
            "PHL",
            "LGA",
        ]
        cities = [
            "San Francisco",
            "New York",
            "Los Angeles",
            "Chicago",
            "Dallas",
            "Denver",
            "Seattle",
            "Atlanta",
            "Miami",
            "Boston",
            "Phoenix",
            "Houston",
            "Las Vegas",
            "Orlando",
            "Newark",
            "Charlotte",
            "Minneapolis",
            "Detroit",
            "Philadelphia",
            "LaGuardia",
        ]
        return json.dumps(dict(zip(airports, cities, strict=False)))

    @staticmethod
    def get_info() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_all_airports",
                "description": "List all airports and their cities.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
