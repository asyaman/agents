"""
Data loaders for the retail benchmark.

Provides functions to load retail data (orders, products, users) and customer personas.
"""

import json
import os
import random
import re
from functools import lru_cache
from typing import Any, Iterator

import loguru
from pydantic import BaseModel, EmailStr

FOLDER_PATH = os.path.dirname(__file__)


@lru_cache(maxsize=1)
def load_data() -> dict[str, Any]:
    """Load retail data from JSON files (cached after first call)."""
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


def clear_data_cache() -> None:
    """Clear the data cache (useful for testing or data updates)."""
    load_data.cache_clear()
    get_all_personas.cache_clear()


class Persona(BaseModel):
    """Represents a customer persona with contact info and order history."""

    name: str
    zip: str
    email: EmailStr
    user_id: str
    orders: list[tuple[str, str]]
    products: list[tuple[str, float]]

    @classmethod
    def from_line(cls, line: str) -> "Persona":
        """Parse a persona from a text line."""
        name = re.search(r"Name: ([^,]+)", line).group(1)  # pyright: ignore[reportOptionalMemberAccess]
        zip_code = re.search(r"Zip: (\d+)", line).group(1)  # pyright: ignore[reportOptionalMemberAccess]
        email = re.search(r"Email: ([^,]+)", line).group(1)  # pyright: ignore[reportOptionalMemberAccess]
        user_id = re.search(r"User_id: ([^,]+)", line).group(1)  # pyright: ignore[reportOptionalMemberAccess]

        orders_raw = re.search(r"Orders: (.*?), Products:", line).group(1)  # pyright: ignore[reportOptionalMemberAccess]
        orders = re.findall(r"\('([^']+)', '([^']+)'\)", orders_raw)

        products_raw = line.split("Products: ")[1].strip().rstrip(".")
        products = re.findall(r"\('([^']+)', ([\d.]+)\)", products_raw)
        products_parsed = [(name, float(price)) for name, price in products]

        return cls(
            name=name,
            zip=zip_code,
            email=email,
            user_id=user_id,
            orders=orders,
            products=products_parsed,
        )


def _load_persona_lines() -> list[str]:
    """Load raw persona lines from file."""
    with open(os.path.join(FOLDER_PATH, "personas.txt")) as file:
        return [line.rstrip() for line in file]


@lru_cache(maxsize=1)
def get_all_personas() -> tuple[Persona, ...]:
    """Get all personas as Pydantic models (cached after first call)."""
    personas: list[Persona] = []
    for line in _load_persona_lines():
        try:
            personas.append(Persona.from_line(line.strip()))
        except AttributeError:
            loguru.logger.warning(
                f"Skipping persona due to parsing error: {line.strip()[:50]}..."
            )
    return tuple(personas)


def get_persona_by_index(index: int) -> Persona:
    """Get a single persona by index."""
    return get_all_personas()[index]


def get_persona_by_email(email: str) -> Persona | None:
    """Find a persona by email address."""
    for persona in get_all_personas():
        if persona.email == email:
            return persona
    return None


def get_persona_by_user_id(user_id: str) -> Persona | None:
    """Find a persona by user ID."""
    for persona in get_all_personas():
        if persona.user_id == user_id:
            return persona
    return None


def get_random_persona() -> Persona:
    """Get a random persona."""
    return random.choice(get_all_personas())


def iter_personas() -> Iterator[Persona]:
    """
    Iterate over all personas one at a time.
    """
    yield from get_all_personas()


def get_persona_count() -> int:
    """Get the total number of personas."""
    return len(get_all_personas())
