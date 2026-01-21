"""Simpler inspection focused on Giana relationship."""
from falkordb import FalkorDB

db = FalkorDB(host='localhost', port=6379)
g = db.select_graph('lcr_memories')

print("=== CHECKING GIANA RELATIONSHIP ===\n")

# Get all relationships
result = g.query("MATCH (a)-[r]->(b) WHERE a.name = 'Jeffrey Kistler' OR b.name = 'Giana' RETURN a.name, type(r), b.name, r.status")
print(f"Query returned {len(result.result_set)} results\n")

for row in result.result_set:
    print(f"{row[0]} --[{row[1]}]--> {row[2]}")
    if len(row) > 3 and row[3]:
        print(f"  Status: {row[3]}")
    print()

# Check for birthday
print("\n=== CHECKING BIRTHDAY ===\n")
result = g.query("MATCH (n) WHERE n.name = 'Jeffrey Kistler' RETURN n.birthday, n.birthdate, n.age")
if result.result_set:
    row = result.result_set[0]
    print(f"Birthday: {row[0]}")
    print(f"Birthdate: {row[1]}")
    print(f"Age: {row[2]}")
else:
    print("No birthday data found")
