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
struct PrefixKey<D>
//	where
//		D: Borrow<[u8]>,
{
	// ([u8; size], next_slice)
	start: u8, // mask of first byte
	end: u8, // mask of last byte
	data: D,
}

impl<D1, D2> PartialEq<PrefixKey<D2>> for PrefixKey<D1>
	where
		D1: Borrow<[u8]>,
		D2: Borrow<[u8]>,
{
	fn eq(&self, other: &PrefixKey<D2>) -> bool {
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
				||(self.unchecked_first_byte() == other.unchecked_first_byte()
					&& self.unchecked_last_byte() == other.unchecked_last_byte()
					&& left[1..left.len() - 1]
						== right[1..right.len() - 1]
			))
	}
}

impl<D> Eq for PrefixKey<D>
	where
		D: Borrow<[u8]>,
{ }


struct Position {
	index: usize,
	mask: u8,
}
impl Position {
	fn zero() -> Self {
		Position {
			index: 0,
			mask: 255,
		}
	}
}

impl<D> PrefixKey<D>
	where
		D: Borrow<[u8]> + Default,
{
	fn empty() -> Self {
		PrefixKey {
			start: 0,
			end: 0,
			data: Default::default(),
		}
	}
}

impl<D> PrefixKey<D>
	where
		D: Borrow<[u8]>,
{

	fn unchecked_first_byte(&self) -> u8 {
		self.data.borrow()[0] & self.start
	}
	fn unchecked_last_byte(&self) -> u8 {
		self.data.borrow()[self.data.borrow().len() - 1] & self.end
	}

	fn pos_start(&self) -> Position {
		Position {
			index: 0,
			mask: self.start,
		}
	}
/*
	fn pos_end(&self) -> Position {
		Position {
			index: self.data.borrow().len(),
			mask: self.end,
		}
	}
*/

	// TODO remove that??
	fn common_depth(&self, other: &Self) -> Position {
		// key must be aligned.
		assert!(self.start == other.start);
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
				mask: 0,
			}
		} else {
			let mask = 255u8 >> delta.leading_zeros();
			Position {
				index,
				mask,
			}
		}
	}

	fn common_depth_next(&self, other: &Self) -> Descent {
		// key must be aligned.
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
		}
	}

	// TODO remove that??
	fn index(&self, ix: Position) -> KeyIndex {
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
}

impl PrefixKey<Vec<u8>>
{
	fn new_offset<Q: Borrow<[u8]>>(key: Q, start: Position) -> Self {
		let data = key.borrow()[start.index..].to_vec();
/*		if data.len() > 0 {
			data[0] &= start.mask; // this update is for Eq implementation
		}*/
		PrefixKey {
			start: start.mask,
			end: 255,
			data,
		}
	}
}

#[derive(PartialEq, Eq, Debug)]
struct Node {
	// TODO this should be able to use &'a[u8] for iteration
	// and querying.
	pub key: PrefixKey<Vec<u8>>,
	//pub value: usize,
	pub value: Option<Vec<u8>>,
	//pub left: usize,
	//pub right: usize,
	// TODO if backend behind, then Self would neeed to implement a Node trait with lazy loading...
	pub children: Children<Self>,
}

impl Node {
	fn leaf(key: &[u8], start: Position, value: Vec<u8>) -> Self {
		Node {
			key: PrefixKey::new_offset(key, start),
			value: Some(value),
			children: Children::empty(),
		}
	}
}

#[derive(PartialEq, Eq, Debug)]
pub struct PrefixMap {
	//tree: Vec<Node>,
	tree: Option<Node>,
	//values: Vec<Vec<u8>>,
	//keys: Vec<u8>,
}

impl PrefixMap {
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

struct KeyIndex { right: bool }
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

enum Descent {
	// index in input key
	Child(Position, KeyIndex),
	Middle(Position, KeyIndex),
	Match(Position),
//	// position mask left of this node
//	Middle(usize, u8),
}

impl Node {
	fn prefix_node(&self, key: &[u8]) -> (&Self, Descent) {
		unimplemented!()
	}
	fn prefix_node_mut(&mut self, key: &[u8]) -> (&mut Self, Descent) {
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
