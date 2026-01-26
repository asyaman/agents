# Copyright Sierra

from typing import Any, Dict, List

from pydantic import BaseModel


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    annotator: str
    user_id: str
    actions: List[Action]
    instruction: str
    outputs: List[str]
