# MCP Local Agent Setup for Qwen 27B

## What This Gives You

An AI agent running on your computer that can:
- Run terminal commands (build, test, install packages)
- Read/write files in your projects  
- Browse and interact with Chrome
- Access databases and APIs

## Installation Steps

### Step 1: Install Open WebUI (Ollama Frontend + MCP)

Open a terminal and run:
```powershell
npm install -g open-webui
open-webui
```

Or if you prefer Docker Desktop (recommended for full features):
```powershell
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway ghcr.io/open-webui/open-webui:main
```

Open browser to http://localhost:3000

### Step 2: Configure MCP Servers in Open WebUI

1. Click Settings (gear icon) → Tools → MCP Servers
2. Add these servers:

#### Filesystem Access
- Name: `filesystem`
- Transport: stdio
- Command: `npx`
- Args: `-y @modelcontextprotocol/server-filesystem C:\Users\clopt\Documents\coding`

#### Shell Commands (Custom)
- Name: `shell`  
- Transport: stdio
- Command: `node`
- Args: `C:\Users\clopt\Documents\coding\Personal_Projects\danielshort3.github.io\.mcp\shell-server.js`

### Step 3: Connect to Qwen Model

In Open WebUI Settings → Models:
- Add model: `qwen2.5:27b` via Ollama provider
- Set context window: 32768 (edit `C:\Users\clopt\AppData\Local\Ollama\ollama.json`)

### Step 4: Use Your Agent

Start a chat and try:
> "Show me the files in my project"
> "Run npm test in the website directory"  
> "Open Chrome and go to localhost:3000"
> "Read my vercel.json and tell me what routes exist"

## Available Tools

| Tool | What It Does | Server |
|------|-------------|--------|
| `execute_command` | Run any PowerShell/terminal command | shell-server.js |
| `read_file` | Read file contents | shell-server.js |
| `list_directory` | List files in a folder | shell-server.js |
| Filesystem tools | Read/write/edit any file in your coding dir | @modelcontextprotocol/server-filesystem |

## How It Works

```
Your Chat Prompt
       |
       v
+------------------+     +---------------------+
|   Open WebUI      |---->|  Ollama Qwen 27B    |
| (Chat Frontend)   |     |  (AI Brain)         |
+------------------+     +----------+----------+
                                      |
                    MCP Protocol       v
                    (stdio JSON-RPC) +---------------------+
                                   |     MCP Servers       |
                                   | - shell-server.js     |
                                   | - filesystem server   |
                                   +---------------------+
                                           |
                                    Your Computer
                                    (Files, Terminal, etc)
```

## Security Notes

- The shell command executor runs on YOUR machine with YOUR permissions
- You can add safety checks to shell-server.js to limit what commands run
- Consider adding a file allowlist/denylist for filesystem access
- All MCP communication stays local (no cloud API calls)

## Troubleshooting

### Shell server fails to start
Check that Node.js is in your PATH: `node --version`

### Ollama model not found
Restart Ollama: `ollama serve` (or restart tray app)

### MCP connection errors
Verify stdio transport works:
```powershell
node .mcp/shell-server.js < nul | timeout /t 3
```

## Next Steps

- Add more MCP servers (GitHub API, Chrome control, database access)
- Create custom tools for your specific workflow
- Set up file watching + auto-rebuild when you make changes
