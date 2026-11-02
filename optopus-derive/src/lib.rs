//! Derive macros for optopus. Use them through `optopus`, which re-exports
//! each one beside the trait it implements.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Expr, Fields, Ident, Index, Meta, Token, parse_macro_input};

/// Implements `optopus::trait_defs::Rankable`.
///
/// With no `rank` attribute the order is the objective, and the type must
/// implement `Evaluate`. The impl compares the two objectives through
/// `optopus::trait_defs::better_by_objective`.
///
/// With `rank` attributes the order compares keys one after another, the first
/// that differs deciding. The keys are named in one of two places, never both.
///
/// - On fields, `#[rank(min)]` or `#[rank(max)]`, compared in the order the
///   fields are declared.
/// - On the type, `#[rank(min = violation(), max = value)]`, compared in the
///   order written. A key is a field, or a method that takes only `&self`.
///
/// `min` ranks the lower key better and `max` the higher one. A key needs only
/// `PartialOrd`, and two keys that are neither less nor greater count as equal.
///
/// The `Rankable` page of the optopus documentation has examples.
#[proc_macro_derive(Rankable, attributes(rank))]
pub fn derive_rankable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match rankable(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Implements `optopus::trait_defs::Evaluate` from one field.
///
/// The field is marked `#[objective(minimize)]` or `#[objective(maximize)]`,
/// and `evaluate` returns it as an `f64` in `Evaluable::Minimize` or
/// `Evaluable::Maximize`. On a solution the field is its objective, and on a
/// move it is the change the move would make. Exactly one field is marked.
///
/// The `Evaluate` page of the optopus documentation has examples.
#[proc_macro_derive(Evaluate, attributes(objective))]
pub fn derive_evaluate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match evaluate(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Implements `optopus::trait_defs::EnabledTabu` from the keys a move touches.
///
/// The keys are named on the type, `#[tabu(dense(i), dense(j))]` for example,
/// and the move is enabled when every key is free and forbids every key when
/// applied, in the order written. Each key reads fields of the move, and each
/// field is a `usize`.
///
/// - `dense(f)` is `TabuKey::DenseVar(f)`, an index bounded by the instance.
/// - `var(f)` is `TabuKey::Var(f)`, an index of any size.
/// - `pair(a, b)` is `TabuKey::Pair(a, b)`, so `(a, b)` and `(b, a)` differ.
/// - `unordered(a, b)` is `TabuKey::Pair(min, max)`, so they are the same.
/// - `triple(a, b, c)` is `TabuKey::Triple(a, b, c)`.
///
/// A move that checks other keys than it forbids implements the trait by hand.
#[proc_macro_derive(EnabledTabu, attributes(tabu))]
pub fn derive_enabled_tabu(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match enabled_tabu(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn enabled_tabu(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let mut keys = Vec::new();
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("tabu")) {
        let entries =
            attr.parse_args_with(syn::punctuated::Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for entry in entries {
            keys.push(tabu_key(&entry)?);
        }
    }
    if keys.is_empty() {
        return Err(syn::Error::new(
            input.ident.span(),
            "name the keys this move touches, as in `#[tabu(dense(i))]`",
        ));
    }
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::optopus::trait_defs::EnabledTabu for #name #ty_generics #where_clause {
            #[inline]
            fn is_move_enabled(
                &self,
                tabu: &::optopus::building_blocks::search::TabuMemory,
                iteration: u64,
            ) -> bool {
                #(tabu.is_enabled(#keys, iteration))&&*
            }

            #[inline]
            fn add_to_tabu_map(
                &self,
                tabu: &mut ::optopus::building_blocks::search::TabuMemory,
                iteration: u64,
                rng: &mut ::optopus::rand::rngs::SmallRng,
            ) {
                #(tabu.forbid(#keys, iteration, rng);)*
            }
        }
    })
}

/// One key of `#[tabu(..)]` as the `TabuKey` expression it builds from `self`.
fn tabu_key(entry: &Meta) -> syn::Result<TokenStream2> {
    let Meta::List(list) = entry else {
        return Err(syn::Error::new(
            entry.span(),
            "expected a key such as `dense(i)` or `pair(i, j)`",
        ));
    };
    let shape = list.path.require_ident()?;
    let fields: Vec<syn::Member> = list
        .parse_args_with(syn::punctuated::Punctuated::<syn::Member, Token![,]>::parse_terminated)?
        .into_iter()
        .collect();
    let arity = |n: usize| -> syn::Result<()> {
        if fields.len() == n {
            Ok(())
        } else {
            Err(syn::Error::new(
                list.span(),
                format!("`{shape}` takes {n} field(s), found {}", fields.len()),
            ))
        }
    };
    let key = quote!(::optopus::building_blocks::search::TabuKey);
    match shape.to_string().as_str() {
        "dense" => {
            arity(1)?;
            let a = &fields[0];
            Ok(quote!(#key::DenseVar(self.#a)))
        }
        "var" => {
            arity(1)?;
            let a = &fields[0];
            Ok(quote!(#key::Var(self.#a)))
        }
        "pair" => {
            arity(2)?;
            let (a, b) = (&fields[0], &fields[1]);
            Ok(quote!(#key::Pair(self.#a, self.#b)))
        }
        "unordered" => {
            arity(2)?;
            let (a, b) = (&fields[0], &fields[1]);
            Ok(quote!(#key::Pair(self.#a.min(self.#b), self.#a.max(self.#b))))
        }
        "triple" => {
            arity(3)?;
            let (a, b, c) = (&fields[0], &fields[1], &fields[2]);
            Ok(quote!(#key::Triple(self.#a, self.#b, self.#c)))
        }
        other => Err(syn::Error::new(
            shape.span(),
            format!("expected `dense`, `var`, `pair`, `unordered` or `triple`, found `{other}`"),
        )),
    }
}

/// Implements `optopus::trait_defs::Distance` from one field.
///
/// The field is marked `#[distance(hamming)]`, and `distance` counts the
/// positions at which the two values of that field differ, through
/// `optopus::building_blocks::representation::hamming_distance`. The field is
/// a slice or anything that derefs to one, such as a `Vec`. Exactly one field
/// is marked.
#[proc_macro_derive(Distance, attributes(distance))]
pub fn derive_distance(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match distance(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn evaluate(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let (field, keyword) = marked_field(input, "objective")?;
    let variant = match keyword.to_string().as_str() {
        "minimize" => quote!(Minimize),
        "maximize" => quote!(Maximize),
        other => {
            return Err(syn::Error::new(
                keyword.span(),
                format!("expected `minimize` or `maximize`, found `{other}`"),
            ));
        }
    };
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::optopus::trait_defs::Evaluate for #name #ty_generics #where_clause {
            #[inline]
            fn evaluate(&self) -> ::optopus::trait_defs::Evaluable<f64> {
                ::optopus::trait_defs::Evaluable::#variant(self.#field as f64)
            }
        }
    })
}

fn distance(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let (field, keyword) = marked_field(input, "distance")?;
    if keyword != "hamming" {
        return Err(syn::Error::new(
            keyword.span(),
            format!("expected `hamming`, found `{keyword}`"),
        ));
    }
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::optopus::trait_defs::Distance for #name #ty_generics #where_clause {
            #[inline]
            fn distance(&self, other: &Self) -> usize {
                ::optopus::building_blocks::representation::hamming_distance(
                    &self.#field,
                    &other.#field,
                )
            }
        }
    })
}

/// The one field of a struct carrying `#[attr(keyword)]`, as the expression
/// that reaches it from `self`, and the keyword.
fn marked_field(input: &DeriveInput, attr: &str) -> syn::Result<(TokenStream2, Ident)> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            input.ident.span(),
            format!("`#[{attr}(..)]` marks a field, so this derive needs a struct"),
        ));
    };
    let mut found: Option<(TokenStream2, Ident)> = None;
    for (i, field) in data.fields.iter().enumerate() {
        for a in field.attrs.iter().filter(|a| a.path().is_ident(attr)) {
            if found.is_some() {
                return Err(syn::Error::new(
                    a.span(),
                    format!("only one field can be marked `#[{attr}(..)]`"),
                ));
            }
            let keyword = a.parse_args::<Ident>()?;
            let access = match &field.ident {
                Some(ident) => quote!(#ident),
                None => {
                    let index = Index::from(i);
                    quote!(#index)
                }
            };
            found = Some((access, keyword));
        }
    }
    found.ok_or_else(|| {
        syn::Error::new(
            input.ident.span(),
            format!("mark the field this derive reads with `#[{attr}(..)]`"),
        )
    })
}

#[derive(Clone, Copy)]
enum Direction {
    Min,
    Max,
}

impl Direction {
    fn parse(ident: &Ident) -> syn::Result<Self> {
        match ident.to_string().as_str() {
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            other => Err(syn::Error::new(
                ident.span(),
                format!("expected `min` or `max`, found `{other}`"),
            )),
        }
    }
}

/// One key, as the code that reads it from `self` and from `other`.
struct Key {
    direction: Direction,
    on_self: TokenStream2,
    on_other: TokenStream2,
}

fn rankable(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let container = container_keys(input)?;
    let fields = field_keys(input)?;

    let body = match (container, fields) {
        (Some((span, _)), Some(_)) => {
            return Err(syn::Error::new(
                span,
                "name the keys either on the type or on its fields, not both",
            ));
        }
        (Some((_, keys)), None) | (None, Some(keys)) => compare_keys(&keys),
        (None, None) => {
            let mut where_clause = where_clause
                .cloned()
                .unwrap_or_else(|| syn::parse_quote!(where));
            where_clause
                .predicates
                .push(syn::parse_quote!(Self: ::optopus::trait_defs::Evaluate));
            return Ok(quote! {
                impl #impl_generics ::optopus::trait_defs::Rankable for #name #ty_generics #where_clause {
                    #[inline]
                    fn is_better_than(&self, other: &Self) -> bool {
                        ::optopus::trait_defs::better_by_objective(self, other)
                    }
                }
            });
        }
    };

    Ok(quote! {
        impl #impl_generics ::optopus::trait_defs::Rankable for #name #ty_generics #where_clause {
            #[inline]
            fn is_better_than(&self, other: &Self) -> bool {
                #body
            }
        }
    })
}

/// The comparison of `keys` in order, the first that differs deciding.
fn compare_keys(keys: &[Key]) -> TokenStream2 {
    let steps = keys.iter().enumerate().map(|(i, key)| {
        let a = format_ident!("__rank_self_{i}");
        let b = format_ident!("__rank_other_{i}");
        let on_self = &key.on_self;
        let on_other = &key.on_other;
        let (better, worse) = match key.direction {
            Direction::Min => (quote!(#a < #b), quote!(#a > #b)),
            Direction::Max => (quote!(#a > #b), quote!(#a < #b)),
        };
        quote! {
            let #a = #on_self;
            let #b = #on_other;
            if #better {
                return true;
            }
            if #worse {
                return false;
            }
        }
    });
    quote! {
        #(#steps)*
        false
    }
}

/// The keys named on the type, with the span of the first `rank` attribute.
fn container_keys(input: &DeriveInput) -> syn::Result<Option<(proc_macro2::Span, Vec<Key>)>> {
    let mut found: Option<(proc_macro2::Span, Vec<Key>)> = None;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("rank")) {
        let entries = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::MetaNameValue, Token![,]>::parse_terminated,
        )?;
        let keys = found.get_or_insert_with(|| (attr.span(), Vec::new()));
        for entry in entries {
            let direction = Direction::parse(entry.path.require_ident()?)?;
            let (on_self, on_other) = key_access(&entry.value)?;
            keys.1.push(Key {
                direction,
                on_self,
                on_other,
            });
        }
    }
    Ok(found)
}

/// How a key named on the type is read, a field or a method taking `&self`.
fn key_access(value: &Expr) -> syn::Result<(TokenStream2, TokenStream2)> {
    match value {
        Expr::Path(path) if path.qself.is_none() => {
            let field = path.path.require_ident()?;
            Ok((quote!(&self.#field), quote!(&other.#field)))
        }
        Expr::Call(call) if call.args.is_empty() => match &*call.func {
            Expr::Path(path) if path.qself.is_none() => {
                let method = path.path.require_ident()?;
                Ok((quote!(self.#method()), quote!(other.#method())))
            }
            other => Err(syn::Error::new(other.span(), "expected a method name")),
        },
        other => Err(syn::Error::new(
            other.span(),
            "expected a field, or a method that takes only `&self`, such as `violation()`",
        )),
    }
}

/// The keys named on the fields, in declaration order.
fn field_keys(input: &DeriveInput) -> syn::Result<Option<Vec<Key>>> {
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) | Data::Union(_) => {
            return Ok(None);
        }
    };
    let mut keys = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        for attr in field.attrs.iter().filter(|a| a.path().is_ident("rank")) {
            let Meta::List(list) = &attr.meta else {
                return Err(syn::Error::new(
                    attr.span(),
                    "expected `#[rank(min)]` or `#[rank(max)]`",
                ));
            };
            let direction = Direction::parse(&list.parse_args::<Ident>()?)?;
            let access = match (&field.ident, fields) {
                (Some(ident), _) => quote!(#ident),
                (None, Fields::Unnamed(_)) => {
                    let index = Index::from(i);
                    quote!(#index)
                }
                (None, _) => unreachable!("a field without a name is in a tuple struct"),
            };
            keys.push(Key {
                direction,
                on_self: quote!(&self.#access),
                on_other: quote!(&other.#access),
            });
        }
    }
    Ok((!keys.is_empty()).then_some(keys))
}
