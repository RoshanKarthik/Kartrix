import { readFileSync } from "node:fs";

export interface Product {
  id: string;
  name: string;
  priceCents: number;
  description: string;
}

export type Catalog = Map<string, Product>;

/** Load the products from a JSON file (default: data/products.json). */
export function loadCatalog(file: URL = new URL("../data/products.json", import.meta.url)): Catalog {
  const products = JSON.parse(readFileSync(file, "utf8")) as Product[];
  return new Map(products.map((p) => [p.id, p]));
}

export function getProduct(catalog: Catalog, id: string): Product | undefined {
  return catalog.get(id);
}
