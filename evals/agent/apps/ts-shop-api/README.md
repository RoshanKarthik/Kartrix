# shop-api

Catalog, cart and pricing for a small shop. TypeScript, no dependencies — Node runs the `.ts`
files directly (type stripping, Node >= 23.6).

```
npm start        # http://127.0.0.1:3000
npm test         # node --test
```

| Method | Path                 | Body / query                          |
|--------|----------------------|---------------------------------------|
| GET    | /products            |                                       |
| GET    | /products/:id        |                                       |
| POST   | /cart/items          | `{"productId": "p1", "quantity": 2}`  |
| GET    | /cart                | `?coupon=SAVE10`                      |

Prices are in cents. Tax (8%) is charged on the subtotal after the coupon discount.
