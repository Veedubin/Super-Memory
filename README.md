# Super-Memory

A semantic memory storage and retrieval MCP (Model Context Protocol) server using LanceDB and sentence transformers.

## What is Super-Memory?

Super-Memory gives your AI agents long-term memory across sessions. It stores and retrieves information using semantic embeddings, so agents can recall relevant context from previous conversations, files, and web pages.

## Features

- **Semantic search** - Query memories by meaning, not just keywords
- **File memory** - Read and store local file contents
- **Web memory** - Fetch and store web page contents
- **Boomerang context** - Special support for Boomerang Protocol session state
- **Local storage** - All data stays on your machine in `./memory_data`

## Tools

| Tool | Description |
|------|-------------|
| `save_to_memory` | Store text with optional metadata |
| `save_file_memory` | Read a file and store its content |
| `save_web_memory` | Fetch a URL and store its content |
| `query_memory` | Semantic search across all memories |
| `list_sources` | List all stored sources |
| `recall_source` | Retrieve exact source by path |
| `save_boomerang_context` | Save Boomerang session context |
| `get_boomerang_context` | Retrieve Boomerang session context |

## Installation

### Using `uv` (recommended)

```bash
uv tool install super-memory
```

### Using `pip`

```bash
pip install super-memory
```

### Manual / Development

```bash
git clone https://github.com/Veedubin/super-memory.git
cd super-memory
uv sync
uv run super-memory
```

## OpenCode Configuration

Add to your `.opencode/opencode.json`:

```json
{
  "mcp": {
    "super-memory": {
      "type": "local",
      "command": ["uv", "run", "super-memory"],
      "enabled": true
    }
  }
}
```

Or if installed with `uv tool`:

```json
{
  "mcp": {
    "super-memory": {
      "type": "local",
      "command": ["super-memory"],
      "enabled": true
    }
  }
}
```

## Requirements

- Python >= 3.12
- CUDA (optional but recommended) - falls back to CPU automatically
- ~500MB disk space for the embedding model (downloaded on first run)

## First Run

On first startup, Super-Memory will download the `BAAI/bge-large-en-v1.5` sentence transformer model. This may take a few minutes depending on your internet connection.

## Data Storage

Memories are stored locally in a `./memory_data` directory relative to where you run the command. Each project should ideally run Super-Memory from its own directory to keep project-specific memories separate.

## License

MIT License - see [LICENSE](LICENSE)
