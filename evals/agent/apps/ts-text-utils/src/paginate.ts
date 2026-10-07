export interface Page<T> {
  items: T[];
  page: number;
  pageCount: number;
}

/** Page ``page`` (from 1) of ``items``; pages past the end are empty. */
export function paginate<T>(items: T[], page: number, perPage: number): Page<T> {
  if (!Number.isInteger(page) || page < 1) throw new RangeError("page must be a positive integer");
  if (!Number.isInteger(perPage) || perPage < 1) throw new RangeError("perPage must be a positive integer");
  const start = (page - 1) * perPage;
  return { items: items.slice(start, start + perPage), page, pageCount: Math.ceil(items.length / perPage) };
}
