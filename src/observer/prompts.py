UTILITY_PROMPT = """Rate the memory-worthiness of this conversation turn.

TURN:
{text}

Rules:
- DISCARD: greetings, thanks, and small talk with no actionable info
- LOW: general discussion but no new facts
- MEDIUM: contains preferences or feelings
- HIGH: contains schedules, relationships, or concrete facts

Respond with exactly one word: DISCARD, LOW, MEDIUM, or HIGH
"""

SUMMARY_PROMPT = """Summarize this conversation turn in a single sentence that highlights what was discussed.

TURN:
{text}

ONE SENTENCE SUMMARY:"""

QUERIES_PROMPT = """List 2-3 questions this turn could answer later. Output a JSON array of strings.

TURN:
{text}

OUTPUT:"""

EXTRACTION_PROMPT = """Extract entities and relationships from the conversation turn.

TURN:
{text}

Instructions:
1. Extract ALL people, places, organizations, and objects mentioned
2. Capture attributes in the "attributes" field (age, role, occupation, etc.)
3. Identify relationships between entities using these types:
   - Familial: SIBLING_OF, PARENT_OF, CHILD_OF, SPOUSE_OF, etc.
   - Social: FRIEND_OF, DATING, MARRIED_TO, BROKE_UP_WITH, etc.
   - Professional: WORKS_AT, MANAGES, COLLEAGUE_OF, etc.
   - Spatial: LIVES_IN, VISITING, TRAVELING_TO, TRAVELING_FROM, etc.
   - Ownership: OWNS, HAS, etc.
   - Emotional: FEELS_ABOUT, PREFERS, DISLIKES, etc.
4. When extracting statements from the user, the subject should be "User"
5. Extract concrete facts only - do not infer or hallucinate

Output valid JSON:
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "RELATIONSHIP_TYPE", "object": "entity2", "metadata": {{"key": "value"}}}}
    ]
}}

Example:
USER: My sister Sarah is 25 and lives in Boston.
ASSISTANT: That's nice!

Would extract:
{{
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{}}}},
        {{"name": "Sarah", "type": "Person", "attributes": {{"age": 25}}}},
        {{"name": "Boston", "type": "Place", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "SIBLING_OF", "object": "Sarah", "metadata": {{"relation": "sister"}}}},
        {{"subject": "Sarah", "predicate": "LIVES_IN", "object": "Boston", "metadata": {{}}}}
    ]
}}

Now extract from the TURN above:"""
