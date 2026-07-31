//! BDD scenarios for the shared checkout flow.

use std::path::PathBuf;

use cucumber::{World, given, then, when};
use ltk_core::checkout::{CartItem, CheckoutResult, checkout_cart};

#[derive(Debug, Default, World)]
struct CheckoutWorld {
    items: Vec<CartItem>,
    result: Option<CheckoutResult>,
}

#[given(expr = "the cart contains {string} priced at {int} cents with quantity {int}")]
async fn cart_contains(world: &mut CheckoutWorld, name: String, price_cents: u32, quantity: u32) {
    world.items.push(CartItem { name, price_cents, quantity });
}

#[when("the customer checks out")]
async fn checkout(world: &mut CheckoutWorld) {
    world.result = Some(checkout_cart(&world.items));
}

#[then(expr = "an order should be created with total {int} cents")]
async fn order_total(world: &mut CheckoutWorld, total_cents: u32) {
    assert!(world.result.is_some());

    if let Some(result) = world.result.as_ref() {
        assert_eq!(result.order.total_cents, total_cents);
    }
}

#[then("the cart should be empty")]
async fn cart_empty(world: &mut CheckoutWorld) {
    assert!(world.result.is_some());

    if let Some(result) = world.result.as_ref() {
        assert!(result.cart.items.is_empty());
    }
}

#[tokio::main]
async fn main() {
    let feature_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../features");
    CheckoutWorld::run(feature_path.as_path()).await;
}
