/** A URL slug: lower-case ASCII letters and digits separated by single dashes, no dash at either end. */
export function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "-")
    .replace(/^-+|-+$/g, "");
}
