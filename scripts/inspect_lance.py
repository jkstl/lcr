import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.vector_store import init_vector_store

table = init_vector_store("./data/lancedb_demo")
print([name for name in dir(table) if not name.startswith("_")])
