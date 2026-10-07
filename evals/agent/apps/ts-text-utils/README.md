# text-utils

Small string helpers, no dependencies.

- `slugify(title)` — URL slug: lower-case ASCII letters and digits separated by single dashes
  (accents are removed: "Crème brûlée" → "creme-brulee").
- `parseDuration("1h30m")` — seconds; units `d`, `h`, `m`, `s`, `ms` (fractions of a second are rounded).
- `paginate(items, page, perPage)` — one page plus the page count.

`npm test` runs the tests (`node --test`).
