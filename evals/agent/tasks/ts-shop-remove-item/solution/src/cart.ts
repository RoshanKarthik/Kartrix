import type { Product } from "./catalog.ts";
import type { PricedLine } from "./pricing.ts";

export interface CartLine extends PricedLine {
  productId: string;
  name: string;
}

export class Cart {
  private lines: Map<string, CartLine> = new Map();

  add(product: Product, quantity: number): CartLine {
    const existing = this.lines.get(product.id);
    const line: CartLine = existing
      ? { ...existing, quantity: existing.quantity + quantity }
      : { productId: product.id, name: product.name, quantity, unitPriceCents: product.priceCents };
    this.lines.set(product.id, line);
    return line;
  }

  /** False if the product had no line in the cart. */
  remove(productId: string): boolean {
    return this.lines.delete(productId);
  }

  list(): CartLine[] {
    return [...this.lines.values()];
  }
}
