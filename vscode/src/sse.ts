/** Server-Sent Events, parsed incrementally from text chunks (the run feed of `GET /v1/runs/{id}/events`). */

export interface SseMessage {
  id: number | null; // the feed's sequence number; null for live-only items (streamed answer tokens)
  kind: string; // event · approval_needed · plan_review_needed · run_finished
  data: Record<string, unknown>;
}

export class SseParser {
  private buffer = "";

  /** Feed a chunk; returns the messages it completed. Comments (`: ping`) and malformed data are skipped. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const messages: SseMessage[] = [];
    let end: number;
    while ((end = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      const message = parseBlock(block);
      if (message) messages.push(message);
    }
    return messages;
  }
}

function parseBlock(block: string): SseMessage | null {
  let id: number | null = null;
  let kind = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "id" && /^\d+$/.test(value)) id = Number(value);
    else if (field === "event") kind = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  try {
    const parsed: unknown = JSON.parse(data.join("\n"));
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    return { id, kind, data: parsed as Record<string, unknown> };
  } catch {
    return null;
  }
}
