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
