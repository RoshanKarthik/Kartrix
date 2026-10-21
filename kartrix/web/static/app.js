/** Kartrix in the browser: sign in, pick or create a workspace, chat with the agent and watch its live feed. */

import { Api, ApiError, parseInput } from "./api.js";
import { Feed, h } from "./feed.js";

const TOKEN_KEY = "kartrix.token";
const root = document.getElementById("app");

const state = {
  api: new Api(),
  info: null, // GET /v1/health: allow_signup, allow_local_paths
  user: null,
  workspaces: [],
  workspace: null,
  runId: null, // the run whose feed is followed
  following: null, // its AbortController
  newSession: false,
};

// ── the token: per-origin browser storage (see kartrix/web/__init__.py for why not a cookie) ──

function loadToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function saveToken(token, remember) {
  try {
    (remember ? localStorage : sessionStorage).setItem(TOKEN_KEY, token);
  } catch {
    // storage blocked: signed in for this page only
  }
}

function clearToken() {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // nothing stored
  }
}

// ── helpers ──

function button(label, onClick, cls = "") {
  const b = h("button", { type: "button", class: cls }, label);
  b.addEventListener("click", onClick);
  return b;
}

function field(labelText, input, hint = null) {
  const id = input.id;
  if (hint) input.setAttribute("aria-describedby", `${id}-hint`);
  return h("div", { class: "field" }, h("label", { for: id }, labelText), input, hint ? h("div", { class: "hint", id: `${id}-hint` }, hint) : null);
}

function message(e) {
  return e instanceof Error ? e.message : String(e);
}

function showFatal(text, retry) {
  stopFollowing();
  root.replaceChildren(
    h("main", { class: "center" }, h("div", { class: "card panel" }, h("h1", {}, "Kartrix"), h("p", { class: "err" }, text), button("Retry", retry, "primary"))),
  );
}

// ── start ──

async function boot() {
  root.replaceChildren(h("main", { class: "center" }, h("p", { class: "dim" }, "Connecting…")));
  state.api = new Api();
  try {
    state.info = await state.api.health();
  } catch (e) {
    showFatal(message(e), boot);
    return;
  }
  const token = loadToken();
  if (token) {
    state.api = new Api(token);
    try {
      state.user = await state.api.me();
      await showMain();
      return;
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 401)) {
        showFatal(message(e), boot);
        return;
      }
      clearToken();
    }
  }
  showSignIn();
}

// ── sign-in ──

function showSignIn(note = "") {
  stopFollowing();
  state.api = new Api();
  state.user = null;
  state.workspace = null;
  document.title = "Kartrix";
  let signup = false;
  const email = h("input", { id: "email", type: "email", autocomplete: "username", required: "", maxlength: "320" });
  const password = h("input", { id: "password", type: "password", autocomplete: "current-password", required: "", minlength: "8", maxlength: "256" });
  const remember = h("input", { id: "remember", type: "checkbox" });
  const title = h("h1", {}, "Sign in to Kartrix");
  const error = h("p", { class: "err", role: "alert" });
  const submit = h("button", { type: "submit", class: "primary wide" }, "Sign in");
  const toggle = button("No account yet? Create one", () => {
    signup = !signup;
    title.textContent = signup ? "Create a Kartrix account" : "Sign in to Kartrix";
    submit.textContent = signup ? "Create account" : "Sign in";
    toggle.textContent = signup ? "Have an account? Sign in" : "No account yet? Create one";
    password.autocomplete = signup ? "new-password" : "current-password";
    error.textContent = "";
  }, "link");
  const form = h(
    "form",
    { class: "card panel auth", novalidate: "" },
    h("div", { class: "brand" }, h("img", { src: "/assets/kartrix.svg", alt: "", width: "28", height: "28" }), h("span", {}, "Kartrix")),
    title,
    note ? h("p", { class: "warn" }, note) : null,
    field("Email", email),
    field("Password", password, "At least 8 characters. The same account works for kartrix login and VS Code."),
    h("label", { class: "check" }, remember, " Keep me signed in on this browser"),
    error,
    submit,
    state.info.allow_signup ? h("p", { class: "switch" }, toggle) : h("p", { class: "dim small" }, "Sign-up is closed on this server — ask its admin for an account."),
  );
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    error.textContent = "";
    if (!email.value.trim() || !password.value) {
      error.textContent = "Enter your email and password.";
      return;
    }
    submit.disabled = true;
    try {
      const api = new Api();
      const result = signup ? await api.signup(email.value.trim(), password.value) : await api.login(email.value.trim(), password.value);
      saveToken(result.token, remember.checked);
      state.api = new Api(result.token);
      state.user = result.user;
      await showMain();
    } catch (err) {
      error.textContent = message(err);
      submit.disabled = false;
    }
  });
  root.replaceChildren(h("main", { class: "center" }, form));
  email.focus();
}

async function signOut() {
  try {
    await state.api.logout();
  } catch {
    // the token is forgotten here either way
  }
  clearToken();
  history.replaceState(null, "", location.pathname);
  showSignIn();
}

/** A failed call: a 401 means the sign-in expired (or was revoked) — back to the sign-in form. */
function failed(e) {
  if (e instanceof ApiError && e.status === 401) {
    clearToken();
    showSignIn("Your sign-in has expired — sign in again.");
    return;
  }
  ui.feed.line(message(e), "err");
}

// ── the main screen ──

const ui = {};

async function showMain() {
  ui.workspaceList = h("ul", { class: "workspaces" });
  ui.workspaceTitle = h("h2", { class: "workspace-title" }, "");
  ui.workspacePath = h("div", { class: "dim small path" });
  ui.newConversation = button("New conversation", newConversation, "small");
  ui.log = h("div", { class: "log", role: "log", "aria-live": "polite" });
  ui.feed = new Feed(ui.log);
  ui.input = h("textarea", {
    id: "request",
    rows: "3",
    "aria-label": "Request",
    placeholder: "Ask about the code or request a change — Enter sends, Shift+Enter starts a new line",
  });
  ui.send = button("Send", submit, "primary");
  ui.stop = button("Stop", stopRun, "danger");
  ui.stop.hidden = true;
  ui.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    }
  });
  ui.chat = h(
    "section",
    { class: "chat" },
    h("div", { class: "chat-head" }, h("div", { class: "chat-title" }, ui.workspaceTitle, ui.workspacePath), ui.newConversation),
    ui.log,
    h("div", { class: "composer" }, ui.input, h("div", { class: "actions" }, ui.stop, ui.send)),
  );
  ui.empty = h(
    "section",
    { class: "chat empty" },
    h("div", { class: "card panel" }, h("h2", {}, "Create a workspace to start"), h("p", { class: "dim" }, state.info.allow_local_paths
      ? "A workspace is a project folder on this machine. Kartrix reads it, answers questions about it and changes it for you."
      : "A workspace holds a project's files on this server. Kartrix reads them, answers questions and changes them for you.")),
  );
  const menu = button("☰", () => document.body.classList.toggle("nav-open"), "icon menu");
  menu.setAttribute("aria-label", "Workspaces");
  const header = h(
    "header",
    { class: "topbar" },
    menu,
    h("div", { class: "brand" }, h("img", { src: "/assets/kartrix.svg", alt: "", width: "22", height: "22" }), h("span", {}, "Kartrix")),
    h("div", { class: "spacer" }),
    h("span", { class: "dim small email", title: "Signed in" }, state.user.email),
    button("Sign out", signOut, "small"),
  );
  ui.main = h("div", { class: "main" }, h("aside", { class: "sidebar" }, h("h2", {}, "Workspaces"), ui.workspaceList, newWorkspaceForm()), ui.empty);
  root.replaceChildren(header, ui.main);
  try {
    state.workspaces = await state.api.workspaces();
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) failed(e);
    else showFatal(message(e), boot);
    return;
  }
  renderWorkspaces();
  const wanted = new URLSearchParams(location.hash.slice(1)).get("w");
  const ws = state.workspaces.find((w) => w.id === wanted) ?? state.workspaces[0];
  if (ws) await selectWorkspace(ws);
  else document.body.classList.add("nav-open");
}

function renderWorkspaces() {
  ui.workspaceList.replaceChildren(
    ...state.workspaces.map((ws) => {
      const pick = button("", () => selectWorkspace(ws), "workspace");
      pick.append(h("span", { class: "name" }, ws.name), state.info.allow_local_paths ? h("span", { class: "path" }, ws.root_path) : null);
      pick.title = state.info.allow_local_paths ? ws.root_path : ws.name;
      if (state.workspace?.id === ws.id) pick.setAttribute("aria-current", "true");
      return h("li", {}, pick);
    }),
  );
  if (!state.workspaces.length) ui.workspaceList.append(h("li", { class: "dim small" }, "No workspaces yet."));
}

function newWorkspaceForm() {
  const local = state.info.allow_local_paths;
  const name = h("input", { id: "ws-name", type: "text", required: "", maxlength: "200", autocomplete: "off" });
  const path = h("input", { id: "ws-path", type: "text", maxlength: "4096", autocomplete: "off", spellcheck: "false", placeholder: "D:\\code\\my-app or /home/me/my-app" });
  const error = h("p", { class: "err small", role: "alert" });
  const create = h("button", { type: "submit", class: "primary" }, "Create");
  const form = h(
    "form",
    { class: "new-workspace" },
    h("h3", {}, "New workspace"),
    field("Name", name),
    local ? field("Folder", path, "An existing folder on this machine (absolute path).") : h("p", { class: "dim small" }, "Its files live on this server."),
    error,
    create,
  );
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    error.textContent = "";
    if (!name.value.trim()) {
      error.textContent = "Give the workspace a name.";
      return;
    }
    if (local && !path.value.trim()) {
      error.textContent = "Enter the project's folder.";
      return;
    }
    create.disabled = true;
    try {
      const ws = await state.api.createWorkspace(name.value.trim(), local ? path.value.trim() : null);
      state.workspaces.push(ws);
      name.value = "";
      path.value = "";
      renderWorkspaces();
      await selectWorkspace(ws);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) failed(err);
      else error.textContent = message(err);
    } finally {
      create.disabled = false;
    }
  });
  return form;
}

async function selectWorkspace(ws) {
  stopFollowing();
  document.body.classList.remove("nav-open");
  state.workspace = ws;
  state.newSession = false;
  history.replaceState(null, "", `#w=${ws.id}`);
  renderWorkspaces();
  if (ui.main.lastChild !== ui.chat) ui.main.replaceChild(ui.chat, ui.main.lastChild);
  ui.workspaceTitle.textContent = ws.name;
  ui.workspacePath.textContent = state.info.allow_local_paths ? ws.root_path : "";
  document.title = `${ws.name} · Kartrix`;
  ui.feed.clear();
  let runs;
  try {
    runs = await state.api.runs(ws.id, 20);
  } catch (e) {
    failed(e);
    return;
  }
  if (state.workspace !== ws) return; // another workspace was picked meanwhile
  let session = null;
  let running = null;
  for (const run of [...runs].reverse()) {
    if (session !== null && run.session_id !== session) ui.feed.line("— new conversation —", "dim divider");
    session = run.session_id;
    if (run.status === "running") {
      ui.feed.user(run.input, run.kind);
      running = run;
    } else {
      ui.feed.past(run);
    }
  }
  if (!runs.length) ui.feed.hint("No requests yet — ask about the code or request a change.");
  ui.log.scrollTop = ui.log.scrollHeight;
  if (running) void follow(running.id);
  ui.input.focus();
}

// ── runs ──

function setBusy(busy) {
  ui.send.disabled = busy;
  ui.stop.hidden = !busy;
  ui.newConversation.disabled = busy;
}

async function submit() {
  const ws = state.workspace;
  if (!ws || state.runId) return;
  const parsed = parseInput(ui.input.value);
  if (parsed.error) {
    ui.feed.line(parsed.error, "warn");
    return;
  }
  ui.input.value = "";
  ui.feed.user(parsed.input, parsed.kind);
  setBusy(true);
  let run;
  try {
    run = await state.api.startRun(ws.id, parsed.kind, parsed.input, { newSession: state.newSession });
  } catch (e) {
    setBusy(false);
    failed(e);
    return;
  }
  state.newSession = false;
  if (state.workspace === ws) await follow(run.id);
}

async function follow(runId) {
  const controller = new AbortController();
  state.runId = runId;
  state.following = controller;
  setBusy(true);
  let reported = false; // the core's own run_finished event already said how it ended
  try {
    for await (const item of state.api.events(runId, controller.signal)) {
      if (item.kind === "run_finished") {
        if (!reported) ui.feed.ended(String(item.data.status ?? "error"), typeof item.data.detail === "string" ? item.data.detail : null);
      } else {
        reported ||= item.kind === "event" && item.data.type === "run_finished";
        ui.feed.message(item);
      }
    }
  } catch (e) {
    if (!controller.signal.aborted) failed(e);
  } finally {
    if (state.following === controller) {
      state.runId = null;
      state.following = null;
      ui.feed.reset();
      setBusy(false);
      ui.input.focus();
    }
  }
}

function stopFollowing() {
  state.following?.abort();
  state.following = null;
  state.runId = null;
  if (ui.feed) {
    ui.feed.reset();
    setBusy(false);
  }
}

async function stopRun() {
  if (!state.runId) return;
  try {
    await state.api.stop(state.runId);
    ui.feed.line("Stopping…", "dim");
  } catch (e) {
    if (!(e instanceof ApiError && (e.status === 404 || e.status === 409))) failed(e); // it ended meanwhile
  }
}

function newConversation() {
  if (state.runId) return;
  state.newSession = true;
  ui.feed.clear();
  ui.feed.line("New conversation — the next request starts fresh.", "dim");
  ui.input.focus();
}

void boot();
