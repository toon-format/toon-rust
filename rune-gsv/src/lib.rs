pub mod addressing;
pub mod slot;
pub mod store;
pub mod persistence;
pub mod builtins;
pub mod query;

pub use crate::addressing::WeylSemanticAddress;
pub use crate::slot::{SGLRuneSlot, SemanticGraph, ExecutionTrace, RankingData};
pub use crate::store::QuantizedContinuum;

pub mod prelude {
    pub use crate::{WeylSemanticAddress, SGLRuneSlot, QuantizedContinuum};
}
