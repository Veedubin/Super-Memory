import lancedb
import torch
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("SuperMemory")

# Local storage path
DB_PATH = "./memory_data"
db = lancedb.connect(DB_PATH)

# Configure the embedding model to use your 4080 Super
# BGE-Large is one of the best open-source embedding models
registry = get_registry().get("sentence-transformers")
embed_model = registry.create(
    name="BAAI/bge-large-en-v1.5", 
    device="cuda" if torch.cuda.is_available() else "cpu"
)

class MemorySchema(LanceModel):
    text: str = embed_model.SourceField()
    vector: Vector(embed_model.ndims()) = embed_model.VectorField()

# Initialize table
table = db.create_table("memories", schema=MemorySchema, exist_ok=True)

@mcp.tool()
def save_to_memory(content: str) -> str:
    """Stores a piece of information into your local long-term memory."""
    table.add([{"text": content}])
    return "Memory archived on GPU."

@mcp.tool()
def query_memory(question: str, top_k: int = 3) -> str:
    """Retrieves relevant past memories based on a semantic search."""
    results = table.search(question).limit(top_k).to_pydantic(MemorySchema)
    
    if not results:
        return "No relevant memories found."
    
    context = "\n---\n".join([r.text for r in results])
    return f"Found these relevant memories:\n\n{context}"

if __name__ == "__main__":
    mcp.run()
