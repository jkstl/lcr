"""
NuExtract templates and examples for LCR entity/relationship extraction.

Templates define the JSON schema that NuExtract uses to guide extraction.
Examples provide in-context learning for better accuracy on complex cases.
"""

from typing import Dict, Any, List


# NuExtract template for entity/relationship extraction
EXTRACTION_TEMPLATE = {
    "fact_type": "string",  # "core", "preference", or "episodic"
    "entities": [
        {
            "name": "verbatim-string",  # Exact name from text
            "type": "string",  # Person, Place, Organization, Technology, Concept, Event
            "attributes": {}  # Additional properties (age, role, etc.)
        }
    ],
    "relationships": [
        {
            "subject": "verbatim-string",  # Entity name (exact)
            "predicate": "string",  # Relationship type
            "object": "verbatim-string",  # Target entity (exact)
            "temporal": "string"  # ongoing, completed, future
        }
    ]
}


# In-context learning examples to improve entity attribution accuracy
EXTRACTION_EXAMPLES = [
    # Example 1: User identity and work
    {
        "input": "USER: My name is Alex Chen and I work at Microsoft as a software engineer.\nASSISTANT: Nice to meet you!",
        "output": """{
  "fact_type": "core",
  "entities": [
    {"name": "User", "type": "Person", "attributes": {}},
    {"name": "Alex Chen", "type": "Person", "attributes": {}},
    {"name": "Microsoft", "type": "Organization", "attributes": {}},
    {"name": "software engineer", "type": "Concept", "attributes": {}}
  ],
  "relationships": [
    {"subject": "User", "predicate": "HAS_NAME", "object": "Alex Chen", "temporal": "ongoing"},
    {"subject": "User", "predicate": "WORKS_AT", "object": "Microsoft", "temporal": "ongoing"}
  ]
}"""
    },
    # Example 2: Named family member with location
    {
        "input": "USER: My sister Emily lives in Seattle and works as a doctor.\nASSISTANT: That's interesting!",
        "output": """{
  "fact_type": "core",
  "entities": [
    {"name": "User", "type": "Person", "attributes": {}},
    {"name": "Emily", "type": "Person", "attributes": {"relation": "sister"}},
    {"name": "Seattle", "type": "Place", "attributes": {}},
    {"name": "doctor", "type": "Concept", "attributes": {}}
  ],
  "relationships": [
    {"subject": "User", "predicate": "SIBLING_OF", "object": "Emily", "temporal": "ongoing"},
    {"subject": "Emily", "predicate": "LIVES_IN", "object": "Seattle", "temporal": "ongoing"},
    {"subject": "Emily", "predicate": "WORKS_AS", "object": "doctor", "temporal": "ongoing"}
  ]
}"""
    },
    # Example 3: User's project work
    {
        "input": "USER: I'm building a weather app called SkyView that uses machine learning.\nASSISTANT: Cool project!",
        "output": """{
  "fact_type": "core",
  "entities": [
    {"name": "User", "type": "Person", "attributes": {}},
    {"name": "SkyView", "type": "Technology", "attributes": {"description": "weather app"}},
    {"name": "machine learning", "type": "Concept", "attributes": {}}
  ],
  "relationships": [
    {"subject": "User", "predicate": "WORKS_ON", "object": "SkyView", "temporal": "ongoing"},
    {"subject": "SkyView", "predicate": "USES", "object": "machine learning", "temporal": "ongoing"}
  ]
}"""
    },
]


def get_extraction_template() -> Dict[str, Any]:
    """
    Get the NuExtract template for entity/relationship extraction.

    Returns:
        JSON template schema
    """
    return EXTRACTION_TEMPLATE


def get_extraction_examples() -> List[Dict[str, str]]:
    """
    Get in-context learning examples for NuExtract extraction.

    These examples help the model understand:
    - Entity attribution (User vs named entities)
    - Family relationships (my sister X -> X is subject, not User)
    - Project relationships (User WORKS_ON project)

    Returns:
        List of example input/output pairs
    """
    return EXTRACTION_EXAMPLES
