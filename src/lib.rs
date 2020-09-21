// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg_attr(not(feature = "std"), no_std)]
// rawvec api handling alloc is quite nice -> TODO manage alloc the same way,
// for now just using vec with usize as ptr
//#![feature(allow_internal_unstable)] 

//! Ordered tree with prefix iterator.
//!
//! Allows iteration over a key prefix.
//! No concern about deletion performance.

// mask cannot be 0 !!! TODO move this in key impl documentation
extern crate alloc;

//use alloc::raw_vec::RawVec;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::borrow::Borrow;
use core::cmp::min;

/*#[cfg(not(feature = "std"))]
extern crate alloc; // TODO check if needed in 2018 and if needed at all

#[cfg(feature = "std")]
mod core_ {
	use alloc::raw_vec::RawVec
}

#[cfg(not(feature = "std"))]
mod core_ {
	pub use core::{borrow, convert, cmp, iter, fmt, hash, marker, mem, ops, result};
	pub use core::iter::Empty as EmptyIter;
	pub use alloc::{boxed, rc, vec};
	pub use alloc::collections::VecDeque;
	pub trait Error {}
	impl<T> Error for T {}
}

#[cfg(feature = "std")]
use self::rstd::{fmt, Error};

use hash_db::MaybeDebug;
use self::rstd::{boxed::Box, vec::Vec};
*/

#[derive(Debug)]
struct PrefixKey<D, P>
	where
		P: PrefixKeyConf,
//		D: Borrow<[u8]>,
{
	// ([u8; size], next_slice)
	start: P::Mask, // mask of first byte
	end: P::Mask, // mask of last byte
	data: D,
}

/// Definition of prefix handle.
pub trait PrefixKeyConf {
	/// Is key byte align using this definition.
	const ALIGNED: bool;
	/// Either u8 or () depending on wether
	/// we use aligned key.
	type Mask: MaskKeyByte;
	/// Index for a given `NodeChildren`.
	type KeyIndex;
	/// Maximum number of children per item.
	const CHILDREN_CAPACITY: usize;
	/// DEPTH in byte when aligned or in bit (2^DEPTH == NUMBER_CHILDREN).
	/// TODO is that of any use?
	const DEPTH: usize;
	/// Advance one item in depth.
	/// Return next mask and true if need to advance index.
	fn advance(previous_mask: Self::Mask) -> (Self::Mask, bool);
	/// Advance with multiple steps.
	fn advance_by(mut previous_mask: Self::Mask, nb: usize) -> (Self::Mask, usize) {
		let mut bytes = 0;
		for _i in 0..nb {
			let (new_mask, b) = Self::advance(previous_mask);
			previous_mask = new_mask;
			if b {
				bytes += 1;
			}
		}
		(previous_mask, bytes)
	}
	/// (get a mask corresponding to a end position).
	// let mask = !(255u8 >> delta.leading_zeros()); + TODO round to nibble
	fn mask_from_delta(delta: u8) -> Self::Mask;
}

/// Mask a byte for unaligned prefix key.
pub trait MaskKeyByte: Eq + core::fmt::Debug {
	fn mask(&self, byte: u8) -> u8;
//	fn mask_mask(&self, other: Self) -> Self;
	fn empty() -> Self;
}

impl MaskKeyByte for () {
	fn mask(&self, byte: u8) -> u8 {
		byte
	}
/*	fn mask_mask(&self, other: Self) -> Self {
		()
	}*/
	fn empty() -> Self {
		()
	}
}

impl MaskKeyByte for u8 {
	fn mask(&self, byte: u8) -> u8 {
		self & byte
	}
/*	fn mask_mask(&self, other: Self) -> Self {
		self & other
	}*/
	fn empty() -> Self {
		0
	}
}

impl<D1, D2, P> PartialEq<PrefixKey<D2, P>> for PrefixKey<D1, P>
	where
		D1: Borrow<[u8]>,
		D2: Borrow<[u8]>,
		P: PrefixKeyConf,
{
	fn eq(&self, other: &PrefixKey<D2, P>) -> bool {
		// !! this means either 255 or 0 mask
		// is forbidden!!
		// 0 should be forbidden, 255 when full byte
		// eg 1 byte slice is 255 and empty is always
		// same as a -1 byte so 255 mask
		let left = self.data.borrow();
		let right = other.data.borrow();
		left.len() == right.len()
			&& self.start == other.start
			&& self.end == other.end
			&& (left.len() == 0
				|| left.len() == 1 && self.unchecked_single_byte() == other.unchecked_single_byte()
				|| (self.unchecked_first_byte() == other.unchecked_first_byte()
					&& self.unchecked_last_byte() == other.unchecked_last_byte()
					&& left[1..left.len() - 1]
						== right[1..right.len() - 1]
			))
	}
}

impl<D, P> Eq for PrefixKey<D, P>
	where
		D: Borrow<[u8]>,
		P: PrefixKeyConf,
{ }

struct Position<P>
	where
		P: PrefixKeyConf,
{
	index: usize,
	mask: P::Mask,
}
impl<P> Position<P>
	where
		P: PrefixKeyConf,
{
	fn zero() -> Self {
		let (mask, next) = P::advance(P::Mask::empty());
		debug_assert!(!next);
		Position {
			index: 0,
			mask,
		}
	}
	fn mask_first() -> P::Mask {
		let (mask, next) = P::advance(P::Mask::empty());
		debug_assert!(!next);
		mask
	}
}

impl<D, P> PrefixKey<D, P>
	where
		D: Borrow<[u8]> + Default,
		P: PrefixKeyConf,
{
	fn empty() -> Self {
		PrefixKey {
			start: P::Mask::empty(),
			end: P::Mask::empty(),
			data: Default::default(),
		}
	}
}

impl<D, P> PrefixKey<D, P>
	where
		D: Borrow<[u8]>,
		P: PrefixKeyConf,
{

	fn unchecked_first_byte(&self) -> u8 {
		self.start.mask(self.data.borrow()[0])
	}
	fn unchecked_last_byte(&self) -> u8 {
		self.end.mask(self.data.borrow()[self.data.borrow().len() - 1])
	}
	fn unchecked_single_byte(&self) -> u8 {
		self.start.mask(self.end.mask(self.data.borrow()[0]))
	}

/*	fn pos_start(&self) -> Position<P> {
		Position {
			index: 0,
			mask: self.start,
		}
	}

	fn pos_end(&self) -> Position {
		Position {
			index: self.data.borrow().len(),
			mask: self.end,
		}
	}
*/

	// TODO remove that??
	fn common_depth(&self, other: &Self) -> Position<P> {
		if P::ALIGNED {
			let mut index = 0;
			let left = self.data.borrow();
			let right = other.data.borrow();
			let upper_bound = min(left.len(), right.len());
			for index in 0..upper_bound {
				if left[index] != right[index] {
					return Position {
						index,
						mask: P::Mask::empty(),
					}
				}
			}
			return Position {
				index: upper_bound,
				mask: P::Mask::empty(),
			}
		}
		if self.start != other.start {
			return Position::zero();
		}
		let left = self.data.borrow();
		let right = other.data.borrow();
		if left.len() == 0 || right.len() == 0 {
			return Position::zero();
		}
		let mut index = 0;
		let mut delta = self.unchecked_first_byte() ^ other.unchecked_last_byte();
		if delta == 0 {
			let upper_bound = min(left.len(), right.len());
			for i in 1..(upper_bound - 1) {
				if left[i] != right[i] {
					index = i;
					break;
				}
			}
			if index == 0 {
				index = upper_bound - 1;
				delta = if left.len() == upper_bound {
					self.unchecked_last_byte() ^ right[index]
				} else {
					left[index] ^ other.unchecked_last_byte()
				};
			} else {
				delta = left[index] ^ right[index];
			}
		}
		if delta == 0 {
			Position {
				index: index + 1,
				mask: P::Mask::empty(),
			}
		} else {
			//let mask = 255u8 >> delta.leading_zeros();
			let mask = P::mask_from_delta(delta);
/*			let mask = if index == 0 {
				self.start.mask_mask(mask)
			} else {
				mask
			};*/
			Position {
				index,
				mask,
			}
		}
	}

	fn common_depth_next(&self, other: &Self) -> Descent<P> {
/*		// key must be aligned.
		assert!(self.start == other.start);
		let left = self.data.borrow();
		let right = other.data.borrow();
		assert!(self.start == other.start);
		if left.len() == 0 {
			if right.len() == 0 {
				return Descent::Match(Position::zero());
			} else {
				return Descent::Middle(Position::zero(), other.index(Position::zero()));
			}
		} else if right.len() == 0 {
			return Descent::Child(Position::zero(), self.index(Position::zero()));
		}
		let mut index = 0;
		let mut delta = self.unchecked_first_byte() ^ other.unchecked_last_byte();
		if delta == 0 {
			let upper_bound = min(left.len(), right.len());
			for i in 1..(upper_bound - 1) {
				if left[i] != right[i] {
					index = i;
					break;
				}
			}
			if index == 0 {
				index = upper_bound - 1;
				delta = if left.len() == upper_bound {
					self.unchecked_last_byte() ^ right[index]
				} else {
					left[index] ^ other.unchecked_last_byte()
				};
			} else {
				delta = left[index] ^ right[index];
			}
		}
		if delta == 0 {
			Position {
				index: index + 1,
				mask: 0,
			}
		} else {
			let mask = 255u8 >> delta.leading_zeros();
			Position {
				index,
				mask,
			}
		}*/
		unimplemented!()
	}
/*
	// TODO remove that??
	fn index(&self, ix: Position<P>) -> P::KeyIndex {
		let mask = 128u8 >> ix.mask.leading_zeros();
		if (self.data.borrow()[ix.index] & mask) == 0 {
			KeyIndex {
				right: false,
			}
		} else {
			KeyIndex {
				right: true,
			}
		}
	}
*/
}

impl<P> PrefixKey<Vec<u8>, P>
	where
		P: PrefixKeyConf,
{
	fn new_offset<Q: Borrow<[u8]>>(key: Q, start: Position<P>) -> Self {
		let data = key.borrow()[start.index..].to_vec();
/*		if data.len() > 0 {
			data[0] &= start.mask; // this update is for Eq implementation
		}*/
		PrefixKey {
			start: start.mask,
			end: P::Mask::empty(),
			data,
		}
	}
}

#[derive(PartialEq, Eq, Debug)]
struct Node<P>
	where
		P: PrefixKeyConf,
{
	// TODO this should be able to use &'a[u8] for iteration
	// and querying.
	pub key: PrefixKey<Vec<u8>, P>,
	//pub value: usize,
	pub value: Option<Vec<u8>>,
	//pub left: usize,
	//pub right: usize,
	// TODO if backend behind, then Self would neeed to implement a Node trait with lazy loading...
	pub children: Children<Self>,
}

impl<P> Node<P>
	where
		P: PrefixKeyConf,
{
	fn leaf(key: &[u8], start: Position<P>, value: Vec<u8>) -> Self {
		Node {
			key: PrefixKey::new_offset(key, start),
			value: Some(value),
			children: Children::empty(),
		}
	}
}

#[derive(PartialEq, Eq, Debug)]
pub struct PrefixMap<P>
	where
		P: PrefixKeyConf,
{
	//tree: Vec<Node>,
	tree: Option<Node<P>>,
	//values: Vec<Vec<u8>>,
	//keys: Vec<u8>,
}

impl<P> PrefixMap<P>
	where
		P: PrefixKeyConf,
{
	pub fn new() -> Self {
		PrefixMap {
			tree: None,
		}
	}

	// TODO should we add value as borrow, skip alloc
	// will see when benching
	pub fn insert(&mut self, key: &[u8], value: Vec<u8>) -> Option<Vec<u8>> {
		if let Some(tree) = self.tree.as_mut() {
			let (parent, descent) = tree.prefix_node_mut(key);
			//	let (prefix, right) = PrefixKey::from(key[key_start..], mask);
			unimplemented!()
		} else {
			self.tree = Some(Node::leaf(key, Position::zero(), value));
			None
		}
	}
}

#[derive(PartialEq, Eq, Debug)]
struct Children<N> {
	left: Option<Box<N>>,
	right: Option<Box<N>>,
}

impl<N> Children<N> {
	fn empty() -> Self {
		Children {
			left: None,
			right: None,
		}
	}
}

enum Descent<P>
	where
		P: PrefixKeyConf,
{
	// index in input key
	Child(Position<P>, P::KeyIndex),
	Middle(Position<P>, P::KeyIndex),
	Match(Position<P>),
//	// position mask left of this node
//	Middle(usize, u8),
}

impl<P> Node<P>
	where
		P: PrefixKeyConf,
{
	fn prefix_node(&self, key: &[u8]) -> (&Self, Descent<P>) {
		unimplemented!()
	}
	fn prefix_node_mut(&mut self, key: &[u8]) -> (&mut Self, Descent<P>) {
		unimplemented!()
	}
}

#[cfg(test)]
mod test {
	use crate::*;

	#[test]
	fn empty_are_equals() {
		let t1 = PrefixMap::new();
		let t2 = PrefixMap::new();
		assert_eq!(t1, t2);
	}

	#[test]
	fn inserts_are_equals() {
		let mut t1 = PrefixMap::new();
		let mut t2 = PrefixMap::new();
		let value1 = b"value1".to_vec();
		assert_eq!(None, t1.insert(b"key1", value1.clone()));
		assert_eq!(None, t2.insert(b"key1", value1.clone()));
		assert_eq!(t1, t2);
		assert_eq!(Some(value1.clone()), t1.insert(b"key1", b"value2".to_vec()));
		assert_eq!(Some(value1.clone()), t2.insert(b"key1", b"value2".to_vec()));
		assert_eq!(t1, t2);
		assert_eq!(None, t1.insert(b"key2", value1.clone()));
		assert_eq!(None, t2.insert(b"key2", value1.clone()));
		assert_eq!(t1, t2);
		assert_eq!(None, t2.insert(b"key3", value1.clone()));
		assert_ne!(t1, t2);
	}
}
