use serde::{Deserialize, Serialize};
use serde_json::json;
use toon_format::encode_default;

#[derive(Debug, Serialize, Deserialize)]
struct SimpleUser {
    id: i32,
    name: String,
    active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct UserInfo {
    id: i32,
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Container {
    user: UserInfo,
}

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: i32,
    name: String,
    email: String,
    active: bool,
}

pub fn objects() {
    // JSON example: Simple object
    let simple = json!({
        "id": 123,
        "name": "Ada",
        "active": true
    });
    let out = encode_default(&simple).unwrap();
    println!("{out}");

    // Struct: Simple object
    let simple_user = SimpleUser {
        id: 123,
        name: "Ada".to_string(),
        active: true,
    };
    let out = encode_default(&simple_user).unwrap();
    println!("\n{out}");

    // JSON example: Nested object
    let nested = json!({
        "user": { "id": 123, "name": "Ada" }
    });
    let out_nested = encode_default(&nested).unwrap();
    println!("\n{out_nested}");

    // Struct: Nested object
    let container = Container {
        user: UserInfo {
            id: 123,
            name: "Ada".to_string(),
        },
    };
    let out = encode_default(&container).unwrap();
    println!("\n{out}");

    // JSON example: Array of objects
    let array_object = json!([
      {
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com",
        "active": true
      },
      {
        "id": 2,
        "name": "Bob",
        "email": "bob@example.com",
        "active": true
      }
    ]);
    let out = encode_default(&array_object).unwrap();
    println!("\n{out}");

    // Struct: Array of objects
    let users = vec![
        User {
            id: 1,
            name: "Alice".to_string(),
            email: "alice@example.com".to_string(),
            active: true,
        },
        User {
            id: 2,
            name: "Bob".to_string(),
            email: "bob@example.com".to_string(),
            active: true,
        },
    ];
    let out = encode_default(&users).unwrap();
    println!("\n{out}");
}
