import { Cart } from "./cart.ts";
import { type Catalog, getProduct } from "./catalog.ts";
import { cartTotal } from "./pricing.ts";

export interface Request {
  method: string;
  path: string; // may include a query string
  body?: unknown;
}

export interface Response {
  status: number;
  body: unknown;
}

export type Handler = (req: Request) => Response;

/** The API as a plain function (src/server.ts serves it over HTTP). One cart per app instance. */
export function createApp(catalog: Catalog): Handler {
  const cart = new Cart();

  return (req) => {
    const url = new URL(req.path, "http://localhost");
    const parts = url.pathname.split("/").filter(Boolean);
    try {
      if (req.method === "GET" && url.pathname === "/products") {
        return { status: 200, body: [...catalog.values()] };
      }
      if (req.method === "GET" && parts[0] === "products" && parts.length === 2) {
        const product = getProduct(catalog, parts[1]);
        if (!product) return { status: 404, body: { error: "product not found" } };
        return { status: 200, body: { id: product.id, name: product.name, priceCents: product.priceCents } };
      }
      if (req.method === "POST" && url.pathname === "/cart/items") {
        const { productId, quantity } = (req.body ?? {}) as { productId?: string; quantity?: number };
        const product = productId ? getProduct(catalog, productId) : undefined;
        if (!product) return { status: 404, body: { error: "unknown product" } };
        const qty = quantity === undefined ? 1 : quantity;
        if (typeof qty !== "number" || !Number.isInteger(qty) || qty < 1 || qty > 99) {
          return { status: 400, body: { error: "quantity must be a whole number from 1 to 99" } };
        }
        return { status: 201, body: cart.add(product, qty) };
      }
      if (req.method === "GET" && url.pathname === "/cart") {
        const lines = cart.list();
        return { status: 200, body: { lines, ...cartTotal(lines, url.searchParams.get("coupon") ?? undefined) } };
      }
      return { status: 404, body: { error: "not found" } };
    } catch (err) {
      return { status: 500, body: { error: "internal error" } };
    }
  };
}
