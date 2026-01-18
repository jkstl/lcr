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

EXTRACTION_PROMPT = """Extract entities, relationships, and classify the fact type from the conversation turn.

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
6. CLASSIFY THE FACT TYPE:
   - "core": Work schedules, recurring routines, family relationships, home address, 
             technology/devices owned, persistent life facts
   - "preference": Opinions, likes/dislikes, feelings, preferences
   - "episodic": One-time events, plans, meetings, trips

Output valid JSON:
{{
    "fact_type": "core|preference|episodic",
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "RELATIONSHIP_TYPE", "object": "entity2", "metadata": {{"key": "value"}}}}
    ]
}}

Example 1 (core fact - work schedule):
USER: I work at TechCorp from 9 to 5 on weekdays.
ASSISTANT: Got it!

Would extract:
{{
    "fact_type": "core",
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{"work_hours": "9-5", "work_days": "weekdays"}}}},
        {{"name": "TechCorp", "type": "Organization", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "WORKS_AT", "object": "TechCorp", "metadata": {{"schedule": "9-5 weekdays"}}}}
    ]
}}

Example 2 (episodic - one-time event):
USER: I'm meeting Sarah for coffee tomorrow at 3pm.
ASSISTANT: Sounds fun!

Would extract:
{{
    "fact_type": "episodic",
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{}}}},
        {{"name": "Sarah", "type": "Person", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "MEETING_WITH", "object": "Sarah", "metadata": {{"time": "3pm", "when": "tomorrow"}}}}
    ]
}}

Example 3 (preference):
USER: I prefer Python over JavaScript for backend work.
ASSISTANT: That's a popular choice!

Would extract:
{{
    "fact_type": "preference",
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{}}}},
        {{"name": "Python", "type": "Technology", "attributes": {{}}}},
        {{"name": "JavaScript", "type": "Technology", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "PREFERS", "object": "Python", "metadata": {{"context": "backend work", "over": "JavaScript"}}}}
    ]
}}

Now extract from the TURN above:"""

