import re
from typing import Any


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Basic regex-driven entity recognition for key patterns."""
    entities: list[dict[str, Any]] = []

    work_match = re.search(r"\bwork at ([A-Za-z0-9 &]+)", text, re.IGNORECASE)
    if work_match:
        entities.append({"name": work_match.group(1).strip(), "type": "Organization", "attributes": {}})

    phone_match = re.search(r"\bphone (?:is|number is) ([\d\w\s\-]+)", text, re.IGNORECASE)
    if phone_match:
        entities.append({"name": phone_match.group(1).strip(), "type": "Technology", "attributes": {}})

    name_match = re.search(r"\bname is ([A-Za-z ]+)", text, re.IGNORECASE)
    if name_match:
        entities.append({"name": name_match.group(1).strip(), "type": "Person", "attributes": {}})

    girlfriend_match = re.search(r"(?:girlfriend|partner) ([A-Za-z]+)", text, re.IGNORECASE)
    if girlfriend_match:
        entities.append({"name": girlfriend_match.group(1).strip(), "type": "Person", "attributes": {"relationship": "partner"}})

    router_match = re.search(r"router(?: that)? can run OpenWRT", text, re.IGNORECASE)
    if router_match:
        entities.append({"name": "OpenWRT-capable router", "type": "Technology", "attributes": {}})

    return entities


def extract_relationships(text: str) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    work_match = re.search(r"\bI work at ([A-Za-z0-9 &]+)", text, re.IGNORECASE)
    if work_match:
        company = work_match.group(1).strip()
        relationships.append(
            {"subject": "User", "predicate": "WORKS_AT", "object": company, "metadata": {}}
        )

    live_match = re.search(r"\blive in ([A-Za-z ]+)", text, re.IGNORECASE)
    if live_match:
        relationships.append(
            {"subject": "User", "predicate": "LIVES_IN", "object": live_match.group(1).strip(), "metadata": {}}
        )

    breakup_match = re.search(r"\bbroke up with my (?:girlfriend|partner) ([A-Za-z]+)", text, re.IGNORECASE)
    if breakup_match:
        name = breakup_match.group(1).strip()
        relationships.append(
            {
                "subject": "User",
                "predicate": "FEELS_ABOUT",
                "object": name,
                "metadata": {"sentiment": "sad", "context": "breakup"},
            }
        )

    router_match = re.search(r"\bnew router that can run OpenWRT", text, re.IGNORECASE)
    if router_match:
        relationships.append(
            {
                "subject": "User",
                "predicate": "PREFERS",
                "object": "OpenWRT-capable router",
                "metadata": {"context": "new hardware", "strength": 0.7},
            }
        )

    return relationships
