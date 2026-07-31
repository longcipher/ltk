//! Checkout utilities (BDD template demo).
//!
//! This module contains shared types and functions for the checkout flow
//! used as a BDD acceptance-test demo.

/// A line item in the shopping cart.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CartItem {
    /// Human-readable item name.
    pub name: String,
    /// Price per unit in cents.
    pub price_cents: u32,
    /// Selected quantity.
    pub quantity: u32,
}

/// A shopping cart.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cart {
    /// Items currently in the cart.
    pub items: Vec<CartItem>,
}

/// An order created from checkout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Order {
    /// Items included in the order.
    pub items: Vec<CartItem>,
    /// Total order value in cents.
    pub total_cents: u32,
}

/// The result of checking out a cart.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckoutResult {
    /// The newly created order.
    pub order: Order,
    /// The emptied cart after checkout.
    pub cart: Cart,
}

/// Builds a greeting string for the CLI.
#[must_use]
pub fn greeting(name: &str) -> String {
    format!("Hello, {name}!")
}

/// Creates an order from the provided items and clears the cart.
#[must_use]
pub fn checkout_cart(items: &[CartItem]) -> CheckoutResult {
    let order_items = items.to_vec();
    let total_cents = order_items.iter().map(|item| item.price_cents * item.quantity).sum();

    CheckoutResult {
        order: Order { items: order_items, total_cents },
        cart: Cart { items: Vec::new() },
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn cart_item_strategy() -> impl Strategy<Value = CartItem> {
        ("[A-Za-z][A-Za-z0-9 ]{0,15}", 0_u16..10_000, 0_u16..100).prop_map(
            |(name, price_cents, quantity)| CartItem {
                name,
                price_cents: u32::from(price_cents),
                quantity: u32::from(quantity),
            },
        )
    }

    #[test]
    fn greeting_builds_message() {
        assert_eq!(greeting("Rust"), "Hello, Rust!");
    }

    #[test]
    fn checkout_cart_creates_an_order_and_clears_the_cart() {
        let result = checkout_cart(&[
            CartItem { name: "Tea".to_string(), price_cents: 450, quantity: 2 },
            CartItem { name: "Cake".to_string(), price_cents: 350, quantity: 1 },
        ]);

        assert_eq!(result.order.total_cents, 1250);
        assert!(result.cart.items.is_empty());
    }

    proptest! {
        #[test]
        fn checkout_cart_preserves_generated_items(
            items in proptest::collection::vec(cart_item_strategy(), 0..16),
        ) {
            let result = checkout_cart(&items);

            prop_assert_eq!(result.order.items, items);
            prop_assert!(result.cart.items.is_empty());
        }

        #[test]
        fn checkout_cart_total_matches_generated_line_items(
            items in proptest::collection::vec(cart_item_strategy(), 0..16),
        ) {
            let expected_total: u32 = items.iter().map(|item| item.price_cents * item.quantity).sum();
            let result = checkout_cart(&items);

            prop_assert_eq!(result.order.total_cents, expected_total);
        }
    }
}
