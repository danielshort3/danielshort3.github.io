// Custom MCP Shell Server - Execute terminal commands
// Usage: node shell-server.js
// Connects via stdio JSON-RPC per MCP spec v2024-11-05

const { stdin, stdout } = process;
let buffer = '';

stdin.on('data', (data) => {
  buffer += data.toString();
  const lines = buffer.split('\n');
  // Process complete JSON-RPC messages
  while (lines.length > 1) {
    const line = lines.shift();
    try {
      const msg = JSON.parse(line);
      if (msg.jsonrpc === '2.0') {
        handleRequest(msg);
      }
    } catch (e) {
      // Incomplete message, wait for more data
      lines.unshift('');
      break;
    }
  }
  buffer = lines.join('\n');
});

async function handleRequest(msg) {
  try {
    if (msg.method === 'initialize') {
      sendResponse(msg.id, {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: 'shell-command-executor', version: '1.0.0' }
      });
    } else if (msg.method === 'tools/list') {
      sendResponse(msg.id, {
        tools: [
          {
            name: 'execute_command',
            description: 'Run a shell command on the local machine. Use for file operations, system info, builds, etc.',
            inputSchema: {
              type: 'object',
              properties: {
                command: { type: 'string', description: 'The shell command to execute' },
                timeout_ms: { type: 'number', description: 'Timeout in milliseconds (default 10000)' }
              },
              required: ['command']
            }
          },
          {
            name: 'read_file',
            description: 'Read contents of a file from the filesystem',
            inputSchema: {
              type: 'object',
              properties: {
                path: { type: 'string', description: 'File path to read' }
              },
              required: ['path']
            }
          },
          {
            name: 'list_directory',
            description: 'List files in a directory',
            inputSchema: {
              type: 'object',
              properties: {
                path: { type: 'string', description: 'Directory path to list' },
                recursive: { type: 'boolean', description: 'List recursively (default false)' }
              },
              required: ['path']
            }
          }
        ]
      });
    } else if (msg.method === 'tools/call') {
      const result = await handleToolCall(msg.params);
      sendResponse(msg.id, result);
    }
  } catch (err) {
    sendError(msg.id, err.message || 'Internal error');
  }
}

async function handleToolCall(params) {
  if (!params || !params.name) throw new Error('Missing tool name');
  
  switch (params.name) {
    case 'execute_command': {
      const cmd = params.arguments?.command;
      const timeout = params.arguments?.timeout_ms || 10000;
      if (!cmd) throw new Error('Command required');
      return await runCommand(cmd, timeout);
    }
    case 'read_file': {
      const path = params.arguments?.path;
      if (!path) throw new Error('Path required');
      const fs = await import('fs/promises');
      try {
        const content = await fs.readFile(path, 'utf8');
        return { content: [{ type: 'text', text: content }] };
      } catch (err) {
        return { isError: true, content: [{ type: 'text', text: `Error reading file: ${err.message}` }] };
      }
    }
    case 'list_directory': {
      const path = params.arguments?.path;
      const recursive = params.arguments?.recursive || false;
      if (!path) throw new Error('Path required');
      const { execSync } = await import('child_process');
      try {
        const result = execSync(`powershell -Command "Get-ChildItem '${path}' ${recursive ? '-Recurse' : ''} | Format-Table"`).toString();
        return { content: [{ type: 'text', text: result }] };
      } catch (err) {
        return { isError: true, content: [{ type: 'text', text: `Error: ${err.message}` }] };
      }
    }
    default:
      throw new Error(`Unknown tool: ${params.name}`);
  }
}

async function runCommand(cmd, timeout) {
  const { execFile } = await import('child_process');
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Command timed out after ${timeout}ms`)), timeout);
    execFile('powershell.exe', ['-Command', cmd], (err, stdout, stderr) => {
      clearTimeout(timer);
      if (err) {
        resolve({ isError: true, content: [{ type: 'text', text: `Error: ${stderr || err.message}` }] });
      } else {
        const output = (stdout || '').trim() + (stderr ? `\nStderr: ${stderr}` : '');
        resolve({ content: [{ type: 'text', text: output || '(No output)' }] });
      }
    });
  });
}

function sendResponse(id, result) {
  const msg = JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n';
  process.stdout.write(msg);
}

function sendError(id, message) {
  const msg = JSON.stringify({ jsonrpc: '2.0', id, error: { code: -32603, message } }) + '\n';
  process.stdout.write(msg);
}

console.error('Shell MCP Server started');
