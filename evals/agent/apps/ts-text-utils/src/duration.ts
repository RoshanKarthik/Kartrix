const UNIT_SECONDS: Record<string, number> = {
  d: 86_400,
  h: 3_600,
  m: 60,
  s: 1,
  ms: 0.001,
};

/** Seconds in a duration like "1h30m", "2d", "45s" or "1500ms" (rounded to whole seconds). */
export function parseDuration(input: string): number {
  const text = input.trim().toLowerCase();
  if (!text) throw new Error("empty duration");
  const pattern = /(\d+(?:\.\d+)?)\s*(ms|d|h|m|s)/y;
  let total = 0;
  let index = 0;
  while (index < text.length) {
    if (text[index] === " ") {
      index++;
      continue;
    }
    pattern.lastIndex = index;
    const match = pattern.exec(text);
    if (!match) throw new Error(`invalid duration: ${input}`);
    const value = Number(match[1]);
    const unit = match[2];
    total += value * UNIT_SECONDS[unit];
    index = pattern.lastIndex;
  }
  return Math.round(total);
}
