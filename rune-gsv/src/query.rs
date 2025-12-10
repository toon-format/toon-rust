use crate::addressing::WeylSemanticAddress;
use crate::slot::SGLRuneSlot;

pub fn count_head_matches(a: &WeylSemanticAddress, b: &SGLRuneSlot) -> usize {
    let bh: [u8; 8] = b.address.clone().try_into().unwrap_or([0u8;8]);
    let wb = WeylSemanticAddress { heads: bh, digest: [0u8; 32], context: None };
    a.matches(&wb)
}
