#[cfg(test)]
mod disabled_tests {
    // Tests compiled when these features are NOT enabled; ensure functions return an error.
    #[cfg(not(feature = "arow"))]
    #[test]
    fn arow_feature_disabled() {
        use rune_cros::arow::*;
        assert!(arrow_to_json(&[]).is_err());
    }

    #[cfg(not(feature = "prqt"))]
    #[test]
    fn prqt_feature_disabled() {
        use rune_cros::prqt::*;
        assert!(parquet_to_json(&[]).is_err());
    }

    #[cfg(not(feature = "npy"))]
    #[test]
    fn npy_feature_disabled() {
        use rune_cros::npy::*;
        assert!(npy_to_json(&[]).is_err());
    }
}
