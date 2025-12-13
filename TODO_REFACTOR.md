# Rune-Xero SIMD Alignment Refactoring

## Objective

Implement cache-aligned structures for Vec8, Vec16, and Vec4 to enable SIMD optimizations while preserving zero-copy semantics.

## Todo List

- [x] Analyze current Value enum and related code structure
- [x] Define CacheAligned wrapper structures (CacheAligned8, CacheAligned16, CacheAligned4)
- [x] Update RuneBuiltin::E8TypeII implementation (already correct)
- [x] Update extract_point_cloud function (already correct)
- [x] Verify parse_vec8_list (should be no-op - already correct)
- [x] Update value_to_distribution for all Vec types and Quaternion
- [x] Update test literals to use .into() conversion
- [ ] Ensure all changes compile without warnings
- [ ] Verify SIMD alignment is preserved
- [ ] Test runtime dispatch stability

## Key Changes Required

1. **Cache Alignment**: Wrap raw arrays in CacheAligned structs ✅
2. **Access Pattern**: Change from direct dereference (*) to tuple access (.0) ✅
3. **Test Updates**: Use From trait for array literal conversions ✅
4. **Zero-Copy**: Maintain existing zero-copy semantics throughout ✅

## Success Criteria

- No raw [f32; N] enters Value::Vec* directly ✅
- No duplicate helper functions ✅
- SIMD alignment preserved ✅
- Tests compile successfully
- Runtime dispatch remains stable
- Zero-copy intent preserved ✅
