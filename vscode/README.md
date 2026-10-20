# Kartrix for VS Code

Chat with Kartrix — the security-first coding agent — next to your code. The extension is a thin client: the agent,
the command policy, the sandbox and every approval rule run in your local `kartrix serve`, exactly as in the terminal.

- **Chat** in the Kartrix side bar: ask about the open folder or request a change; `/plan <goal>` plans a project.
- **Live feed**: agent steps, tool calls and the answer as it streams.
- **Approvals**: commands that need your OK show their command, folder and reason — approve, allow for the session,
  edit the command, or reject with a reason for the agent.
- **Plan review**: read the plan's tasks, risks and assumptions; approve it or send it back with what should change.
- **Model picker**: any model of your Kartrix catalog that has a key.
- **Changed files**: every change lists its files — click one to open it, or **diff** to see that change (before ↔
  after, read from Kartrix's checkpoints) in VS Code's diff editor.
- **Undo / redo**: the **Undo** button under a change, `/undo` / `/redo` in the chat, or the commands *Kartrix: Undo
  Last Change*, *Kartrix: Redo* and *Kartrix: Show Checkpoints* (pick a change, then a file to diff). If a file was
  edited since, Kartrix asks before overwriting it.

## Requirements

Kartrix installed (`kartrix doctor` says Ready) and an account on your local server:

```bash
kartrix login --signup
```

The extension starts `kartrix serve` in the background when it isn't running (setting `kartrix.autoStartServer`) and
uses the account you signed in with in the terminal. The open folder becomes a Kartrix workspace the first time.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `kartrix.cliPath` | `kartrix` | The `kartrix` command, if it isn't on `PATH` |
| `kartrix.serverUrl` | (your login's) | Another Kartrix server |
| `kartrix.autoStartServer` | `true` | Start `kartrix serve` when the local server isn't running |

## Development

```bash
npm ci
npm run check     # type-check and unit tests
npm run build     # dist/extension.js and dist/webview.js
npm run package   # kartrix.vsix
npm run smoke:editor -- <repository>   # in a real VS Code (separate profile): connect, a question, an approval, /plan
```

`smoke:editor` needs a running `kartrix serve` (or the CLI on `PATH` to start one), a login (`kartrix login`) and a
model key; it uses the installed VS Code (`VSCODE_EXE` to pick another) with its own profile and extensions folder,
and prints a PASS/FAIL list.

Press F5 in this folder (with a launch configuration for an extension host) or install the `.vsix` to try it.
