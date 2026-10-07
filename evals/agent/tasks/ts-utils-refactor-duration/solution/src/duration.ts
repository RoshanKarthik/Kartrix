const UNIT_SECONDS: Record<string, number> = {
  d: 86_400,
  h: 3_600,
  m: 60,
  s: 1,
  ms: 0.001,
};

export type DurationToken = [value: number, unit: string];

/** The [value, unit] pairs of a duration like "1h30m"; throws on anything else. */
export function tokenizeDuration(input: string): DurationToken[] {
  const text = input.trim().toLowerCase();
  if (!text) throw new Error("empty duration");
  const pattern = /(\d+(?:\.\d+)?)\s*(ms|d|h|m|s)/y;
  const tokens: DurationToken[] = [];
  let index = 0;
  while (index < text.length) {
    if (text[index] === " ") {
      index++;
      continue;
    }
    pattern.lastIndex = index;
    const match = pattern.exec(text);
    if (!match) throw new Error(`invalid duration: ${input}`);
    tokens.push([Number(match[1]), match[2]]);
    index = pattern.lastIndex;
  }
  return tokens;
}

/** Seconds in a duration like "1h30m", "2d", "45s" or "1500ms" (rounded to whole seconds). */
export function parseDuration(input: string): number {
  const total = tokenizeDuration(input).reduce((sum, [value, unit]) => sum + value * UNIT_SECONDS[unit], 0);
  return Math.round(total);
}
