# AgentTrace

AgentTrace is an observability and attribution tool for AI-generated code.

## Git Extension

This project includes a Git Extension that integrates with version control to track AI edits and provide attribution data.

### Troubleshooting 'git' command not found
If you see an error saying `git` is not recognized, it means Git is installed but not in your system PATH. 

**Quick fix for current session:**
```powershell
$env:Path += ";C:\Program Files\Git\bin"
```

**Permanent fix:**
1. Open the Start Search, type in "env", and choose "Edit the system environment variables".
2. Click the "Environment Variables..." button.
3. Under "System Variables", find the `Path` variable and click "Edit".
4. Click "New" and add `C:\Program Files\Git\bin`.
5. Restart your terminal.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tool:
   ```bash
   python main.py
   ```