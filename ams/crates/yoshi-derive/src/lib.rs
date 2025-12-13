/* yoshi-derive/src/lib.rs */
//!▫~•◦-----------------------‣
//! # Yoshi Derive – A Lightweight, Macro-Based Error Handling Library.
//!▫~•◦----------------------------------------------------------------‣
//!
//! A procedural derive macro for attribute-style error formatting that generates
//! `Display`, `Error`, and `From` implementations for enums. Uses zero-copy parsing
//! of the AST to minimize memory usage during compilation.
//!
//! ## Key Capabilities
//! - **Display strings** per variant via `#[anyerror("…")]` or `#[anyerror(display = "…")]`
//! - **Struct/tuple/unit** variants supported
//! - **`transparent`** single-field variants delegate `Display` and `Error::source()`
//! - **`source`** detection via `#[anyerror(source)]`, `#[source]`, or by naming a field `source`
//! - **`from`** conversions for single-field variants via `#[anyerror(from)]` or `#[from]`
//! - Full support for **generics** and **where-clauses**
//!
//! ### Architectural Notes
//! This macro uses efficient parsing techniques by operating on borrowed references
//! to the input AST instead of creating full copies where possible. It leverages the
//! `syn` crate for AST parsing and the `quote` crate for code generation.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(AnyError, attributes(anyerror, source, from))]
pub fn derive_anyerror(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Delegate to the internal model which handles parsing and generation.
    // This keeps the proc-macro entry point clean and focused on conversion.
    match model::ErrorEnum::from_syn(&input) {
        Ok(model) => model.generate().into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Internal domain model for the derive macro.
/// Encapsulated in a module to avoid exporting types from the proc-macro crate.
/// This model uses borrowed references to the input AST to achieve zero-copy processing.
mod model {
    use proc_macro2::TokenStream;
    use quote::{format_ident, quote};
    use syn::{
        punctuated::Punctuated, token::Comma, Attribute, Data, DeriveInput, Fields, Ident, Lit,
        Meta,
    };

    // --- Domain Model (Zero-Copy) ---

    pub struct ErrorEnum<'a> {
        ident: &'a Ident,
        generics: &'a syn::Generics,
        variants: Vec<ErrorVariant<'a>>,
        global_config: GlobalConfig,
    }

    #[derive(Clone)]
    pub struct GlobalConfig {
        pub auto_backtrace: bool,
        pub auto_location: bool,
    }

    impl Default for GlobalConfig {
        fn default() -> Self {
            Self {
                auto_backtrace: true,
                auto_location: true,
            }
        }
    }

    impl GlobalConfig {
        fn parse(attrs: &[Attribute]) -> syn::Result<Self> {
            let mut config = Self::default();
            for attr in attrs {
                if attr.path().is_ident("anyerror") {
                    // This handles #[anyerror(no_auto_backtrace, ...)]
                    if let Ok(metas) = attr.parse_args_with(Punctuated::<Meta, Comma>::parse_terminated) {
                        for meta in metas {
                            if let Meta::Path(p) = meta {
                                if p.is_ident("no_auto_backtrace") {
                                    config.auto_backtrace = false;
                                } else if p.is_ident("no_auto_location") {
                                    config.auto_location = false;
                                }
                            }
                        }
                    }
                }
            }
            Ok(config)
        }
    }

    struct ErrorVariant<'a> {
        ident: &'a Ident,
        fields: Vec<ErrorField<'a>>,
        config: VariantConfig,
        original: &'a syn::Variant,
    }

    /// The logical role of a field, if any.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FieldRole {
        Source,
        Location,
        Backtrace,
    }

    struct ErrorField<'a> {
        ident: Option<&'a Ident>,
        ty: &'a syn::Type,
        /// Optional single role for fields like `source`, `location`, or `backtrace`.
        role: Option<FieldRole>,
        is_from: bool,
        is_box: bool,
    }

    #[derive(Default)]
    struct VariantConfig {
        display: Option<syn::LitStr>,
        transparent: bool,
        from: bool,
    }

    // --- Implementation ---

    impl<'a> ErrorEnum<'a> {
        pub(crate) fn from_syn(input: &'a DeriveInput) -> syn::Result<Self> {
            let Data::Enum(data_enum) = &input.data else {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "AnyError can only be derived for enums",
                ));
            };

            let variants = data_enum
                .variants
                .iter()
                .map(ErrorVariant::from_syn)
                .collect::<syn::Result<Vec<_>>>()?;

            // Validation: Check for conflicting From implementations by type
            let mut from_types = std::collections::HashMap::new();
            for variant in &variants {
                if variant.config.from || variant.fields.iter().any(|f| f.is_from) {
                    // Find the type that this variant would generate From for
                    let from_type = if variant.fields.iter().any(|f| f.is_from) {
                        // Field-level #[from]
                        variant.fields.iter().find_map(|f| if f.is_from { Some(f.ty) } else { None })
                    } else if variant.config.transparent && variant.fields.len() == 1 {
                        // Transparent single-field
                        Some(variant.fields[0].ty)
                    } else {
                        None // Variant-level #[from] without transparent - no From type
                    };

                    if let Some(ty) = from_type {
                        let type_str = quote::quote!(#ty).to_string();
                        let other_variant = from_types.get(&type_str);
                        if let Some(other_ident) = other_variant {
                            return Err(syn::Error::new_spanned(
                                &input.ident,
                                format!(
                                    "Conflicting `From<{}>` implementations for variants '{}' and '{}'. Only one variant per enum can generate `From` for each type.",
                                    type_str, other_ident, variant.ident
                                ),
                            ));
                        }
                        from_types.insert(type_str, variant.ident.to_string());
                    }
                }
            }

            let global_config = GlobalConfig::parse(&input.attrs)?;

            Ok(Self {
                ident: &input.ident,
                generics: &input.generics,
                variants,
                global_config,
            })
        }

        pub(crate) fn generate(&self) -> TokenStream {
            let name = self.ident;
            let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();

            // --- Bound Generation for Transparent Generics ---
            let mut display_where_clause = where_clause.cloned();
            let mut error_where_clause = where_clause.cloned();

            let generic_params: std::collections::HashSet<_> =
                self.generics.type_params().map(|p| &p.ident).collect();

            for variant in &self.variants {
                if variant.config.transparent {
                    if let Some(field) = variant.fields.first() {
                        if let syn::Type::Path(tp) = field.ty {
                            if let Some(segment) = tp.path.segments.first() {
                                if generic_params.contains(&segment.ident) {
                                    let ty = &segment.ident;
                                    let display_preds = &mut display_where_clause
                                        .get_or_insert_with(|| syn::parse_quote!(where))
                                        .predicates;
                                    display_preds.push(syn::parse_quote!(#ty: ::std::fmt::Display));

                                    let error_preds = &mut error_where_clause
                                        .get_or_insert_with(|| syn::parse_quote!(where))
                                        .predicates;
                                    error_preds.push(syn::parse_quote!(#ty: ::std::error::Error + 'static));
                                }
                            }
                        }
                    }
                }
            }
            // --- End Bound Generation ---

            let display_arms = self.variants.iter().map(ErrorVariant::generate_display);
            let source_arms = self.variants.iter().map(|v| v.generate_source());
            let from_impls = self
                .variants
                .iter()
                .filter_map(|v| v.generate_from(name, self.generics, &self.global_config));

            quote! {
                impl #impl_generics ::std::fmt::Display for #name #ty_generics #display_where_clause {
                    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                        match self {
                            #( #display_arms ),*
                        }
                    }
                }

                impl #impl_generics ::std::error::Error for #name #ty_generics #error_where_clause {
                    fn source(&self) -> ::std::option::Option<&(dyn ::std::error::Error + 'static)> {
                        match self {
                            #( #source_arms ),*
                        }
                    }
                }

                #( #from_impls )*
            }
        }
    }

    impl<'a> ErrorVariant<'a> {
        fn from_syn(v: &'a syn::Variant) -> syn::Result<Self> {
            let config = VariantConfig::parse(&v.attrs)?;
            let fields = v
                .fields
                .iter()
                .map(ErrorField::from_syn)
                .collect::<syn::Result<Vec<_>>>()?;

            // Validation: Transparent
            if config.transparent {
                if fields.len() != 1 {
                    return Err(syn::Error::new_spanned(
                        &v.ident,
                        "transparent variant must have exactly one field",
                    ));
                }
                if config.display.is_some() {
                    return Err(syn::Error::new_spanned(
                        &v.ident,
                        "`transparent` cannot be combined with a display string",
                    ));
                }
            } else if fields.is_empty() && config.display.is_none() {
                // Validation: Unit variant missing display
                return Err(syn::Error::new_spanned(
                    &v.ident,
                    "Missing #[anyerror(\"...\")] or #[anyerror(display = \"...\")] on unit variant",
                ));
            }

            // Validation: Source count
            let source_count = fields
                .iter()
                .filter(|f| matches!(f.role, Some(FieldRole::Source)))
                .count();
            if source_count > 1 {
                return Err(syn::Error::new_spanned(
                    &v.ident,
                    "Multiple `source` fields detected; mark only one `#[anyerror(source)]` or `#[source]`",
                ));
            }

            // Validation: From count
            let from_count = fields.iter().filter(|f| f.is_from).count();
            if from_count > 1 {
                return Err(syn::Error::new_spanned(
                    &v.ident,
                    "Multiple `#[from]` fields detected; mark at most one field with `#[from]`",
                ));
            }

            // Auto-from for transparent
            let mut final_config = config;
            if final_config.transparent {
                final_config.from = true;
            }

            Ok(Self {
                ident: &v.ident,
                fields,
                config: final_config,
                original: v,
            })
        }

        fn generate_display(&self) -> TokenStream {
            let v_ident = self.ident;

            if self.config.transparent {
                let field = &self.fields[0];
                let binding = field
                    .ident
                    .map_or_else(|| quote! { ( inner ) }, |id| quote! { { #id } });
                let access = field.ident.map_or_else(|| quote! { inner }, |id| quote! { #id });
                quote! { Self::#v_ident #binding => ::std::fmt::Display::fmt(#access, f) }
            } else if let Some(fmt) = &self.config.display {
                let pat = self.generate_bind_pattern();
                let args = self.generate_format_args();
                let touch = self.generate_touch_vars();
                quote! {
                    Self::#v_ident #pat => {
                        #touch
                        ::std::write!(f, #fmt #args)
                    }
                }
            } else {
                // Default display for variants without a format string (non-unit only)
                let pat = self.generate_bind_pattern_ignore();
                let name = self.ident.to_string();
                quote! { Self::#v_ident #pat => ::std::write!(f, #name) }
            }
        }

        fn generate_touch_vars(&self) -> TokenStream {
            match &self.original.fields {
                Fields::Named(n) => {
                    let idents = n.named.iter().map(|f| f.ident.as_ref().unwrap());
                    // Create a void use of references to silence unused warnings without #[allow]
                    quote! { let _ = ( #( &#idents ),* ); }
                }
                Fields::Unnamed(u) => {
                    let vars = (0..u.unnamed.len()).map(|i| format_ident!("_arg{}", i));
                    quote! { let _ = ( #( &#vars ),* ); }
                }
                Fields::Unit => quote! {},
            }
        }

        fn generate_source(&self) -> TokenStream {
            let v_ident = self.ident;
            let source_field = self
                .fields
                .iter()
                .find(|f| matches!(f.role, Some(FieldRole::Source)));

            if self.config.transparent {
                let field = &self.fields[0];
                let binding = field
                    .ident
                    .map_or_else(|| quote! { ( inner ) }, |id| quote! { { #id } });
                let access = field.ident.map_or_else(|| quote! { inner }, |id| quote! { #id });

                let source_logic = if matches!(field.role, Some(FieldRole::Source)) {
                    let as_ref = if field.is_box {
                        quote! { .as_ref() }
                    } else {
                        quote! {}
                    };
                    quote! { ::std::option::Option::Some(#access #as_ref as &(dyn ::std::error::Error + 'static)) }
                } else {
                    // Transparent forwarding of source()
                    quote! { ::std::error::Error::source(#access) }
                };

                quote! { Self::#v_ident #binding => #source_logic }
            } else {
                source_field.map_or_else(
                    || {
                        let pat = self.generate_bind_pattern_ignore();
                        quote! { Self::#v_ident #pat => ::std::option::Option::None }
                    },
                    |field| {
                        let (binding, access) = match field.ident {
                            Some(id) => (quote! { { #id, .. } }, quote! { #id }),
                            None => {
                                let idx = self
                                    .fields
                                    .iter()
                                    .position(|f| std::ptr::eq(f as *const _, field as *const _))
                                    .unwrap();
                                let underscores = vec![quote! { _ }; idx];
                                let trailing = vec![quote! { _ }; self.fields.len() - idx - 1];
                                (
                                    quote! { ( #(#underscores,)* source, #(#trailing,)* ) },
                                    quote! { source },
                                )
                            }
                        };
                        let as_ref = if field.is_box {
                            quote! { .as_ref() }
                        } else {
                            quote! {}
                        };
                        quote! { Self::#v_ident #binding => ::std::option::Option::Some(#access #as_ref as &(dyn ::std::error::Error + 'static)) }
                    },
                )
            }
        }

                fn generate_from(
            &self,
            enum_name: &Ident,
            generics: &syn::Generics,
            global_config: &GlobalConfig,
        ) -> Option<TokenStream> {
            let from_field = if self.config.from || self.fields.iter().any(|f| f.is_from) {
                // Find the explicit `#[from]` field, or the single field in a transparent variant.
                self.fields
                    .iter()
                    .find(|f| f.is_from || self.config.transparent)
            } else {
                None
            };

            let Some(source) = from_field else {
                return None;
            };

            // Ensure all other fields can be auto-initialized (location/backtrace).
            // If any field cannot, we cannot safely generate a `From` impl.
            for field in &self.fields {
                if !std::ptr::eq(field, source)
                    && !matches!(field.role, Some(FieldRole::Location | FieldRole::Backtrace))
                {
                    // This variant has fields that cannot be safely initialized, so skip `From` generation.
                    return None;
                }
            }

            let field_ty = source.ty;
            let v_ident = self.ident;

            let ctor = match self.original.fields {
                Fields::Unnamed(_) => {
                    let mut args = Vec::new();
                    for f in &self.fields {
                        if std::ptr::eq(f, source) {
                            args.push(quote! { v });
                        } else if matches!(f.role, Some(FieldRole::Location)) && global_config.auto_location {
                            args.push(quote! { $crate::location!() });
                        } else if matches!(f.role, Some(FieldRole::Backtrace)) && global_config.auto_backtrace {
                            args.push(quote! { ::std::backtrace::Backtrace::capture() });
                        } else {
                            // A field cannot be auto-initialized. Abort `From` generation for this variant.
                            return None;
                        }
                    }
                    quote! { #enum_name::#v_ident( #(#args),* ) }
                }
                Fields::Named(_) => {
                    let source_ident = source.ident.as_ref()?;
                    let mut assignments = Vec::new();
                    for f in &self.fields {
                         if std::ptr::eq(f, source) {
                            assignments.push(quote! { #source_ident: v });
                        } else if matches!(f.role, Some(FieldRole::Location)) && global_config.auto_location {
                            let fid = f.ident.as_ref()?;
                            assignments.push(quote! { #fid: $crate::location!() });
                        } else if matches!(f.role, Some(FieldRole::Backtrace)) && global_config.auto_backtrace {
                             let fid = f.ident.as_ref()?;
                            assignments.push(quote! { #fid: ::std::backtrace::Backtrace::capture() });
                        } else if !std::ptr::eq(f, source) {
                             // This field is not the source and cannot be auto-initialized. Abort.
                             return None;
                        }
                    }

                    quote! {
                        #enum_name::#v_ident {
                           #( #assignments ),*
                        }
                    }
                }
                Fields::Unit => return None,
            };

            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

            Some(quote! {
                impl #impl_generics ::std::convert::From<#field_ty> for #enum_name #ty_generics #where_clause {
                    fn from(v: #field_ty) -> Self {
                        #ctor
                    }
                }
            })
        }

        // --- Helpers ---

        fn generate_bind_pattern(&self) -> TokenStream {
            match &self.original.fields {
                Fields::Named(n) => {
                    let idents = n.named.iter().map(|f| f.ident.as_ref().unwrap());
                    quote! { { #( #idents ),* } }
                }
                Fields::Unnamed(u) => {
                    let vars = (0..u.unnamed.len()).map(|i| format_ident!("_arg{}", i));
                    quote! { ( #( #vars ),* ) }
                }
                Fields::Unit => quote! {},
            }
        }

        fn generate_bind_pattern_ignore(&self) -> TokenStream {
            match &self.original.fields {
                Fields::Named(_) => quote! { { .. } },
                Fields::Unnamed(_) => quote! { ( .. ) },
                Fields::Unit => quote! {},
            }
        }

        fn generate_format_args(&self) -> TokenStream {
            match &self.original.fields {
                Fields::Named(n) => {
                    let Some(display_lit) = &self.config.display else {
                        return quote! {};
                    };

                    let format_str = display_lit.value();
                    let mut used_fields = std::collections::HashSet::new();

                    // Simple but robust parser for `{field}` or `{field:specifier}`
                    let mut chars = format_str.chars().peekable();
                    while let Some(ch) = chars.next() {
                        if ch == '{' {
                            if chars.peek() == Some(&'{') {
                                chars.next(); // Skip escaped `{{`
                                continue;
                            }

                            let mut ident = String::new();
                            while let Some(next_ch) = chars.next() {
                                if next_ch.is_alphanumeric() || next_ch == '_' {
                                    ident.push(next_ch);
                                } else if next_ch == '}' || next_ch == ':' {
                                    break;
                                }
                                // Ignore other chars inside braces for this simple parser
                            }
                            if !ident.is_empty() {
                                used_fields.insert(ident);
                            }
                        }
                    }

                    let args = n
                        .named
                        .iter()
                        .filter_map(|f| f.ident.as_ref())
                        .filter(|id| used_fields.contains(&id.to_string()));

                    quote! { #( , #args = #args )* }
                }
                Fields::Unnamed(u) => {
                    let vars = (0..u.unnamed.len()).map(|i| format_ident!("_arg{}", i));
                    quote! { , #( #vars )* }
                }
                Fields::Unit => quote! {},
            }
        }
    }

    impl VariantConfig {
        fn parse(attrs: &[Attribute]) -> syn::Result<Self> {
            let mut config = Self::default();
            for attr in attrs {
                if attr.path().is_ident("anyerror") {
                    if let Ok(lit) = attr.parse_args::<syn::LitStr>() {
                        config.display = Some(lit);
                        continue;
                    }

                    let metas: Punctuated<Meta, Comma> =
                        attr.parse_args_with(Punctuated::parse_terminated)?;
                    for meta in metas {
                        match meta {
                            Meta::Path(p) if p.is_ident("transparent") => config.transparent = true,
                            Meta::Path(p) if p.is_ident("from") => config.from = true,
                            Meta::NameValue(nv) if nv.path.is_ident("display") => {
                                if let syn::Expr::Lit(el) = nv.value {
                                    if let Lit::Str(s) = el.lit {
                                        config.display = Some(s);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Ok(config)
        }
    }

impl<'a> ErrorField<'a> {
        fn from_syn(f: &'a syn::Field) -> syn::Result<Self> {
            let mut is_source = false;
            let mut is_from = false;
            let mut is_backtrace = false;

            for attr in &f.attrs {
                if attr.path().is_ident("source") {
                    is_source = true;
                } else if attr.path().is_ident("from") {
                    is_from = true;
                } else if attr.path().is_ident("anyerror") {
                    let metas: Punctuated<Meta, Comma> =
                        attr.parse_args_with(Punctuated::parse_terminated)?;
                    for meta in metas {
                        if let Meta::Path(p) = meta {
                            if p.is_ident("source") {
                                is_source = true;
                            }
                            if p.is_ident("from") {
                                is_from = true;
                            }
                            if p.is_ident("backtrace") {
                                is_backtrace = true;
                            }
                        }
                    }
                }
            }

            // Auto-detect source by name if not explicitly marked
            if !is_source {
                if let Some(id) = &f.ident {
                    if id == "source" {
                        is_source = true;
                    }
                }
            }

            // Check if type is Box<T> for source handling
            let is_box = matches!(&f.ty, syn::Type::Path(tp) if tp.path.segments.first().is_some_and(|s| s.ident == "Box"));

            // Auto-detect location by name ("location") or type ("Location")
            let is_location = f.ident.as_ref().map_or(false, |id| id == "location")
                || matches!(&f.ty, syn::Type::Path(tp) if tp.path.segments.last().as_ref().map_or(false, |s| s.ident == "Location"));

            let role = if is_source {
                Some(FieldRole::Source)
            } else if is_location {
                Some(FieldRole::Location)
            } else if is_backtrace {
                Some(FieldRole::Backtrace)
            } else {
                None
            };

            Ok(Self {
                ident: f.ident.as_ref(),
                ty: &f.ty,
                role,
                is_from,
                is_box,
            })
        }
    }
}
