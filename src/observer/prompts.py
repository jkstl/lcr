# System messages for fine-tuned model (LFM2.5-1.2B-Instruct)
# These differentiate between utility grading and entity extraction tasks
UTILITY_SYSTEM_MESSAGE = "You are a utility grading assistant. Rate conversations as DISCARD (no information), STORE (preferences/opinions), or IMPORTANT (identity/relationships/life facts). Respond with exactly one word."

EXTRACTION_SYSTEM_MESSAGE = "You are an entity extraction assistant. Extract entities and relationships from conversations in JSON format with entities (name, type, attributes) and relationships (subject, predicate, object, temporal)."

# Utility grading prompt
UTILITY_PROMPT = """Rate the memory-worthiness of this conversation turn using a simplified 3-level system.

TURN:
{text}

SIMPLE 3-LEVEL SYSTEM:

**DISCARD** - Pure acknowledgments with ZERO information
Examples: "thanks", "ok", "hi", "you're welcome", "sure", "got it"
→ No facts, no preferences, nothing to remember

**STORE** - Any factual information worth remembering  
- User mentions preferences, opinions, likes/dislikes
- User shares thoughts, feelings, or general information
- Any conversation that contains concrete details
- Technical discussions, tool preferences, general facts
→ If the turn has ANY factual content

**IMPORTANT** - Critical life facts that define the user
- User's identity: name, job, company, role, responsibilities
- User's relationships: family, friends, colleagues, romantic partners
- User's situation: where they live, major life events, state changes
- User's possessions: devices they own, tools used daily
- Life events: job changes, moves, relationship changes, major plans
→ Information about WHO the user IS or HOW they LIVE

DECISION PROCESS:
1. Is there ANY content beyond greetings? NO → DISCARD
2. Does it reveal user's IDENTITY or LIFE SITUATION? YES → IMPORTANT  
3. Otherwise → STORE

EXAMPLES:

Input: USER: I work at Acme Corp as a software engineer on the ML team
       ASSISTANT: That's interesting!
Output: IMPORTANT
Why: Defines user's job, company, role, team (core identity)

Input: USER: My sister Justine lives in Worcester Massachusetts  
       ASSISTANT: That's nice!
Output: IMPORTANT
Why: Family relationship + location (critical life fact)

Input: USER: I prefer Python over JavaScript
       ASSISTANT: That's common for backend
Output: STORE
Why: Preference worth remembering, but not core identity

Input: USER: I bought a Dell Latitude 5520 laptop with 32GB RAM
       ASSISTANT: Nice specs!
Output: IMPORTANT
Why: Device ownership (possession user uses regularly)

Input: USER: I've been thinking about Giana a lot lately and it's making me sad
       ASSISTANT: I'm sorry to hear that
Output: IMPORTANT
Why: Emotional state about specific person (relationship status)

Input: USER: I don't like spicy food
       ASSISTANT: Noted!
Output: STORE
Why: Preference worth remembering, but not life-defining

Input: USER: Tell me more about that
       ASSISTANT: Sure, what would you like to know?
Output: STORE
Why: Has minimal content but shows interest/engagement

Input: USER: Thanks!
       ASSISTANT: You're welcome!
Output: DISCARD  
Why: Zero content, pure acknowledgment

Respond with exactly one word: DISCARD, STORE, or IMPORTANT
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
2. Capture attributes in the "attributes" field (age, role, occupation, dates, birthdays, etc.)
   - For dates/birthdays: use "birthday", "birthdate", or "born_on" keys
   - For ages: use "age" key
   - For temporal info in relationships: use metadata field

3. Identify relationships between entities using these types:
   - Identity: HAS_NAME, NICKNAME, GOES_BY (for user's name or preferred name)
   - Familial: SIBLING_OF, PARENT_OF, CHILD_OF, SPOUSE_OF, etc.
   - Social: FRIEND_OF, DATING, MARRIED_TO, BROKE_UP_WITH, etc.
   - Professional: WORKS_AT, MANAGES, COLLEAGUE_OF, etc.
   - Projects: WORKS_ON, DEVELOPING, MAINTAINING, CREATED (for personal/hobby projects)
   - Spatial (ONGOING states): LIVES_IN, VISITING, STAYING_AT, TRAVELING_TO
   - Spatial (COMPLETED states): RETURNED_HOME, LEFT, DEPARTED, ARRIVED_AT, VISITED, MOVED_TO
   - Ownership: OWNS, HAS, etc.
   - Emotional: FEELS_ABOUT, PREFERS, DISLIKES, MISSES, etc.

4. CRITICAL - ENTITY ATTRIBUTION (pay close attention):
   - Use "User" ONLY when extracting facts about the user themselves ("I work at...", "I prefer...")
   - When the user mentions OTHER PEOPLE by name, use THEIR NAME as the subject, NOT "User"
   - Examples:
     * "My sister Justine lives in Boston" → subject="Justine" (NOT "User")
     * "Mom went home to Massachusetts" → subject="Mom" (NOT "User")
     * "I work at Google" → subject="User" (correct, user talking about themselves)

5. CRITICAL - TEMPORAL STATES (pay close attention):
   - "left and went home" = RETURNED_HOME or LEFT (completed state, NOT VISITING)
   - "was visiting, now back home" = RETURNED_HOME (completed state)
   - "arrived at" = ARRIVED_AT (completed state)
   - "scheduled for Friday" = SCHEDULED_FOR (future state)
   - "already happened" = HAPPENED (completed state)
   - "is visiting" / "currently at" = VISITING (ongoing state)

6. Extract concrete facts only - do not infer or hallucinate

7. CLASSIFY THE FACT TYPE:
   - "core": User's name, work schedules, recurring routines, family relationships, home address,
             technology/devices owned, persistent life facts
   - "preference": Opinions, likes/dislikes, feelings, preferences
   - "episodic": One-time events, plans, meetings, trips, state changes

Output valid JSON (MUST use "predicate" not "relation"):
{{
    "fact_type": "core|preference|episodic",
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "RELATIONSHIP_TYPE", "object": "entity2", "metadata": {{"key": "value"}}}}
    ]
}}

Example 1 (User's work - subject is "User"):
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

Example 2 (Family member's location - subject is the family member, NOT User):
USER: My sister Justine lives in Worcester Massachusetts.
ASSISTANT: That's nice!

Would extract:
{{
    "fact_type": "core",
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{}}}},
        {{"name": "Justine", "type": "Person", "attributes": {{"relationship": "sister"}}}},
        {{"name": "Worcester Massachusetts", "type": "Place", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "SIBLING_OF", "object": "Justine", "metadata": {{}}}},
        {{"subject": "Justine", "predicate": "LIVES_IN", "object": "Worcester Massachusetts", "metadata": {{}}}}
    ]
}}

Example 3 (Temporal state - completed action "left and went home"):
USER: My mom and sister Justine left yesterday and went back home to Massachusetts.
ASSISTANT: Safe travels!

Would extract:
{{
    "fact_type": "episodic",
    "entities": [
        {{"name": "Mom", "type": "Person", "attributes": {{}}}},
        {{"name": "Justine", "type": "Person", "attributes": {{"relationship": "sister"}}}},
        {{"name": "Massachusetts", "type": "Place", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "SIBLING_OF", "object": "Justine", "metadata": {{}}}},
        {{"subject": "Mom", "predicate": "RETURNED_HOME", "object": "Massachusetts", "metadata": {{"when": "yesterday"}}}},
        {{"subject": "Justine", "predicate": "RETURNED_HOME", "object": "Massachusetts", "metadata": {{"when": "yesterday"}}}}
    ]
}}

Example 4 (User's preference - subject is "User"):
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

Example 5 (Temporal state - ongoing visit):
USER: My mom is visiting me in Philadelphia this week.
ASSISTANT: Enjoy your time together!

Would extract:
{{
    "fact_type": "episodic",
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{}}}},
        {{"name": "Mom", "type": "Person", "attributes": {{}}}},
        {{"name": "Philadelphia", "type": "Place", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "Mom", "predicate": "VISITING", "object": "Philadelphia", "metadata": {{"when": "this week"}}}}
    ]
}}

Example 6 (Emotional content - subject is "User"):
USER: I've been thinking about Giana a lot lately and it makes me sad.
ASSISTANT: I'm sorry to hear that.

Would extract:
{{
    "fact_type": "preference",
    "entities": [
        {{"name": "User", "type": "Person", "attributes": {{}}}},
        {{"name": "Giana", "type": "Person", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "User", "predicate": "MISSES", "object": "Giana", "metadata": {{"frequency": "lately"}}}},
        {{"subject": "User", "predicate": "FEELS_ABOUT", "object": "Giana", "metadata": {{"emotion": "sad"}}}}
    ]
}}

Now extract from the TURN above:"""

SEMANTIC_CONTRADICTION_PROMPT = """Analyze if a new relationship contradicts existing facts, considering temporal states and semantic meaning.

NEW RELATIONSHIP:
{new_relationship}

EXISTING RELATIONSHIPS (about the same entities):
{existing_relationships}

Instructions:
1. Consider semantic contradictions, not just exact predicate matches
2. Understand temporal state transitions:
   - "VISITING" contradicts "RETURNED_HOME", "LEFT", "DEPARTED"
   - "SCHEDULED_FOR" contradicts "HAPPENED", "CANCELED", "RESCHEDULED"
   - "TRAVELING_TO" contradicts "ARRIVED_AT", "CANCELED_TRIP"
   - "LIVES_IN" contradicts "MOVED_TO", "RELOCATED_TO"
   - "IS_AT" contradicts "LEFT_FROM"
   - "WORKS_AT" (ongoing) contradicts "RESIGNED_FROM", "FIRED_FROM"
3. Understand state completions:
   - Ongoing states (visiting, working, living) are replaced by completed states (visited, resigned, moved)
   - Future plans (scheduled, planning) are replaced by outcomes (happened, canceled)
4. Same entities with mutually exclusive states = contradiction
5. Updates to attributes (age, location, status) = contradiction if different values

Output valid JSON with reasoning:
{{
    "contradictions": [
        {{
            "existing_id": "id of contradicted fact",
            "existing_statement": "subject predicate object",
            "reason": "explanation of why these contradict",
            "temporal_type": "state_completion|mutual_exclusion|attribute_update|null",
            "confidence": "high|medium|low"
        }}
    ]
}}

Examples:

Example 1 - State completion:
NEW: "Mom RETURNED_HOME Massachusetts"
EXISTING: [
    {{"id": "123", "subject": "Mom", "predicate": "VISITING", "object": "Philadelphia"}},
    {{"id": "124", "subject": "Mom", "predicate": "LIVES_IN", "object": "West Boylston"}}
]

Output:
{{
    "contradictions": [
        {{
            "existing_id": "123",
            "existing_statement": "Mom VISITING Philadelphia",
            "reason": "RETURNED_HOME indicates the VISITING state has completed - Mom is no longer visiting",
            "temporal_type": "state_completion",
            "confidence": "high"
        }}
    ]
}}

Example 2 - No contradiction (sequential states):
NEW: "Mom ARRIVED_AT Philadelphia"
EXISTING: [
    {{"id": "125", "subject": "Mom", "predicate": "TRAVELING_TO", "object": "Philadelphia"}}
]

Output:
{{
    "contradictions": []
}}
(TRAVELING_TO → ARRIVED_AT is a natural progression, not a contradiction)

Example 3 - Attribute update:
NEW: "Sister AGE 25"
EXISTING: [
    {{"id": "126", "subject": "Sister", "predicate": "AGE", "object": "24"}}
]

Output:
{{
    "contradictions": [
        {{
            "existing_id": "126",
            "existing_statement": "Sister AGE 24",
            "reason": "User corrected sister's age from 24 to 25",
            "temporal_type": "attribute_update",
            "confidence": "high"
        }}
    ]
}}

Now analyze the relationships above:"""
