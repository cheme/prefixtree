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
use derivative::Derivative;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::borrow::Borrow;
use core::cmp::min;
use core::fmt::Debug;

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

#[derive(Derivative)]
#[derivative(Clone)]
#[derivative(Debug)]
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
}

impl PrefixKeyConf for () {
	const ALIGNED: bool = true;
	type Mask = ();
}

impl PrefixKeyConf for bool {
	const ALIGNED: bool = false;
	type Mask = bool;
}

type MaskFor<N> = <<N as RadixConf>::Alignment as PrefixKeyConf>::Mask;

/// Definition of node handle.
pub trait RadixConf {
	/// Prefix alignement and mask.
	type Alignment: PrefixKeyConf;
	/// Index for a given `NodeChildren`.
	type KeyIndex: NodeIndex;
	/// Maximum number of children per item.
	const CHILDREN_CAPACITY: usize;
	/// DEPTH in byte when aligned or in bit (2^DEPTH == NUMBER_CHILDREN).
	/// TODO is that of any use?
	const DEPTH: usize;
	/// Advance one item in depth.
	/// Return next mask and number of incremented bytes.
	fn advance(previous_mask: MaskFor<Self>) -> (MaskFor<Self>, usize);
	/// Advance with multiple steps.
	fn advance_by(mut previous_mask: MaskFor<Self>, nb: usize) -> (MaskFor<Self>, usize) {
		let mut bytes = 0;
		for _i in 0..nb {
			let (new_mask, b) = Self::advance(previous_mask);
			previous_mask = new_mask;
			bytes += b;
		}
		(previous_mask, bytes)
	}
	/// (get a mask corresponding to a end position).
	// let mask = !(255u8 >> delta.leading_zeros()); + TODO round to nibble
	fn mask_from_delta(delta: u8) -> MaskFor<Self>;

	fn mask_first() -> MaskFor<Self> {
		let (mask, next) = Self::advance(MaskFor::<Self>::empty());
		debug_assert!(next == 0);
		mask
	}
}

type PositionFor<N> = Position<<<N as Node>::Radix as RadixConf>::Alignment>;
type KeyIndexFor<N> = <<N as Node>::Radix as RadixConf>::KeyIndex;

pub trait Node: Clone + PartialEq + Debug {
	type Radix: RadixConf;
	type InitFrom;

	fn new(
		key: &[u8],
		position: PositionFor<Self>,
		value: Option<&[u8]>,
		init: Self::InitFrom,
	) -> Self;
	fn descend(
		&self,
		key: &[u8],
		node_position: PositionFor<Self>,
		dest_position: PositionFor<Self>,
	) -> Descent<Self::Radix>;
	fn value(
		&self,
	) -> Option<&[u8]>; // TODO parameterized with V
	fn get_child(
		&self,
		index: KeyIndexFor<Self>,
	) -> Option<&Self>;
}

pub struct Radix256RadixConf;
pub struct Radix2Conf;

impl RadixConf for Radix256RadixConf {
	type Alignment = ();
	type KeyIndex = u8;
	const CHILDREN_CAPACITY: usize = 256;
	const DEPTH: usize = 1;
	fn advance(_previous_mask: MaskFor<Self>) -> (MaskFor<Self>, usize) {
		((), 1)
	}
	fn advance_by(_previous_mask: MaskFor<Self>, nb: usize) -> (MaskFor<Self>, usize) {
		((), nb)
	}
	fn mask_from_delta(_delta: u8) -> MaskFor<Self> {
		()
	}

	fn mask_first() -> MaskFor<Self> {
		()
	}
}

impl RadixConf for Radix2Conf {
	type Alignment = bool;
	type KeyIndex = bool;
	const CHILDREN_CAPACITY: usize = 2;
	const DEPTH: usize = 1;
	fn advance(previous_mask: MaskFor<Self>) -> (MaskFor<Self>, usize) {
		if previous_mask {
			(false, 1)
		} else {
			(true, 0)
		}
	}
	fn advance_by(_previous_mask: MaskFor<Self>, nb: usize) -> (MaskFor<Self>, usize) {
		unimplemented!()
	}
	fn mask_from_delta(_delta: u8) -> MaskFor<Self> {
		unimplemented!()
	}
	fn mask_first() -> MaskFor<Self> {
		true
	}
}

/// Mask a byte for unaligned prefix key.
pub trait MaskKeyByte: Clone + Copy + PartialEq + Debug {
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

impl MaskKeyByte for bool {
	fn mask(&self, byte: u8) -> u8 {
		if *self {
			byte & 0xf0
		} else {
			byte
		}
	}
	fn empty() -> Self {
		false
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

#[derive(Derivative)]
#[derivative(Clone)]
#[derivative(Copy)]
pub struct Position<P>
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
		Position {
			index: 0,
			mask: P::Mask::empty(),
		}
	}
	fn next<R: RadixConf<Alignment = P>>(&self) -> Self {
		let (mask, increment) = R::advance(self.mask);
		Position {
			index: self.index + increment,
			mask: P::Mask::empty(),
		}
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
}
// TODO remove that??
fn common_depth<D, N>(one: &PrefixKey<D, N::Alignment>, other: &PrefixKey<D, N::Alignment>) -> Position<N::Alignment>
	where
		D: Borrow<[u8]>,
		N: RadixConf,
{
		if N::Alignment::ALIGNED {
			let mut index = 0;
			let left = one.data.borrow();
			let right = other.data.borrow();
			let upper_bound = min(left.len(), right.len());
			for index in 0..upper_bound {
				if left[index] != right[index] {
					return Position {
						index,
						mask: MaskFor::<N>::empty(),
					}
				}
			}
			return Position {
				index: upper_bound,
				mask: MaskFor::<N>::empty(),
			}
		}
		if one.start != other.start {
			return Position::zero();
		}
		let left = one.data.borrow();
		let right = other.data.borrow();
		if left.len() == 0 || right.len() == 0 {
			return Position::zero();
		}
		let mut index = 0;
		let mut delta = one.unchecked_first_byte() ^ other.unchecked_last_byte();
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
					one.unchecked_last_byte() ^ right[index]
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
				mask: MaskFor::<N>::empty(),
			}
		} else {
			//let mask = 255u8 >> delta.leading_zeros();
			let mask = N::mask_from_delta(delta);
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

//	fn common_depth_next(&self, other: &Self) -> Descent<P> {
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
//	}
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

#[derive(Derivative)]
#[derivative(Clone)]
#[derivative(Debug)]
#[derivative(PartialEq)]
struct NodeOld<P, C>
	where
		P: RadixConf,
//		C: Children<Self, Radix = P>,
{
	// TODO this should be able to use &'a[u8] for iteration
	// and querying.
	pub key: PrefixKey<Vec<u8>, P::Alignment>,
	//pub value: usize,
	pub value: Option<Vec<u8>>,
	//pub left: usize,
	//pub right: usize,
	// TODO if backend behind, then Self would neeed to implement a Node trait with lazy loading...
	pub children: C,
}
/*
impl<P, C> NodeOld<P, C>
	where
		P: RadixConf,
		C: Children<Self, Radix = P>,
{
	fn leaf(key: &[u8], start: Position<P::Alignment>, value: Vec<u8>) -> Self {
		NodeOld {
			key: PrefixKey::new_offset(key, start),
			value: Some(value),
			children: C::empty(),
		}
	}
}
*/

impl<P, C> Node for NodeOld<P, C>
	where
		P: RadixConf,
		C: Children<Self, Radix = P>,
{
	type Radix = P;
	type InitFrom = ();
	fn new(
		key: &[u8],
		position: PositionFor<Self>,
		value: Option<&[u8]>,
		_init: Self::InitFrom,
	) -> Self {
		NodeOld {
			key: PrefixKey::new_offset(key, position),
			value: value.map(|v| v.to_vec()),
			children: C::empty(),
		}
	}
	fn descend(
		&self,
		key: &[u8],
		node_position: PositionFor<Self>,
		dest_position: PositionFor<Self>,
	) -> Descent<Self::Radix> {
		unimplemented!()
	}
	fn value(
		&self,
	) -> Option<&[u8]> {
		unimplemented!()
	}
	fn get_child(
		&self,
		index: KeyIndexFor<Self>,
	) -> Option<&Self> {
		unimplemented!()
	}
}

#[derive(Derivative)]
#[derivative(Clone(bound=""))]
#[derivative(Debug(bound=""))]
#[derivative(PartialEq(bound=""))]
pub struct Trie<N>
	where
		N: Node,
{
	//tree: Vec<Node>,
	tree: Option<N>,
	//values: Vec<Vec<u8>>,
	//keys: Vec<u8>,
}

impl<N> Trie<N>
	where
		N: Node,
{
	pub fn new() -> Self {
		Trie {
			tree: None,
		}
	}

	// TODO should we add value as borrow, skip alloc
	// will see when benching
	pub fn insert(&mut self, key: &[u8], value: Vec<u8>) -> Option<Vec<u8>> {
/*		if let Some(tree) = self.tree.as_mut() {
			let (parent, descent) = tree.prefix_node_mut(key);
			//	let (prefix, right) = PrefixKey::from(key[key_start..], mask);
			unimplemented!()
		} else {
			self.tree = Some(NodeOld::leaf(key, Position::zero(), value));
			None
		}*/
		unimplemented!()
	}
}

pub trait Children<N>: Clone + Debug + PartialEq {
	type Radix: RadixConf;

	fn empty() -> Self;
}

pub trait NodeIndex: Clone + Copy + Debug + PartialEq {
	fn zero() -> Self;
	fn next(&self) -> Option<Self>;
}

impl NodeIndex for bool {
	fn zero() -> Self {
		false
	}
	fn next(&self) -> Option<Self> {
		if *self {
			None
		} else {
			Some(true)
		}
	}
}

impl NodeIndex for u8 {
	fn zero() -> Self {
		0
	}
	fn next(&self) -> Option<Self> {
		if *self == 255 {
			None
		} else {
			Some(*self + 1)
		}
	}
}


#[derive(Derivative)]
#[derivative(Clone)]
#[derivative(Debug)]
#[derivative(PartialEq)]
struct Children2<N> (
	Option<Box<(N, N)>>
);

impl<N: Node> Children<N> for Children2<N> {
	type Radix = Radix2Conf;

	fn empty() -> Self {
		Children2(None)
	}
}

#[derive(Derivative)]
#[derivative(Clone)]
struct Children256<N> (
	// 256 array is to big but ok for initial implementation
	Option<Box<[N; 256]>>
);

impl<N: PartialEq> PartialEq for Children256<N> {
	fn eq(&self, other: &Self) -> bool {
		match (self.0.as_ref(), other.0.as_ref()) {
			(Some(self_children), Some(other_children)) =>  {
				for i in 0..256 {
					if self_children[i] != other_children[i] {
						return false;
					}
				}
				true
			},
			(None, None) => true,
			_ => false,
		}
	}
}
impl<N: Debug> Debug for Children256<N> {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
		if let Some(children) = self.0.as_ref() {
			children[..].fmt(f)
		} else {
			let empty: &[N] = &[]; 
			empty.fmt(f)
		}
	}
}

impl<N: Node> Children<N> for Children256<N> {
	type Radix = Radix256RadixConf;

	fn empty() -> Self {
		Children256(None)
	}
}

// TODO macro the specialized impl
#[derive(Derivative)]
#[derivative(Clone)]
struct Children256Bis (
	// 256 array is to big but ok for initial implementation
	Option<Box<[NodeOld<Radix256RadixConf, Children256Bis>; 256]>>
);

impl PartialEq for Children256Bis {
	fn eq(&self, other: &Self) -> bool {
		match (self.0.as_ref(), other.0.as_ref()) {
			(Some(self_children), Some(other_children)) =>  {
				for i in 0..256 {
					if self_children[i] != other_children[i] {
						return false;
					}
				}
				true
			},
			(None, None) => true,
			_ => false,
		}
	}
}
impl Debug for Children256Bis {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
		if let Some(children) = self.0.as_ref() {
			children[..].fmt(f)
		} else {
			let empty: &[NodeOld<Radix256RadixConf, Children256Bis>] = &[]; 
			empty.fmt(f)
		}
	}
}

impl Children<NodeOld<Radix256RadixConf, Children256Bis>> for Children256Bis {
	type Radix = Radix256RadixConf;

	fn empty() -> Self {
		Children256Bis(None)
	}
}

#[derive(Derivative)]
#[derivative(Clone)]
#[derivative(Copy)]
pub enum Descent<P>
	where
		P: RadixConf,
{
	// index in input key
	/// Position is the position of branch, child is at position + 1.
	/// Index is index for the child at position.
	Child(Position<P::Alignment>, P::KeyIndex),
	/// Position is the position where we branch from key.
	/// Index is the index where we can insert child for this key. TODO consider removing
	/// index (we probably need to calculate the current child index too.
	Middle(Position<P::Alignment>, P::KeyIndex),
	/// Position is the position of the node that match.
	Match(Position<P::Alignment>),
//	// position mask left of this node
//	Middle(usize, u8),
}

impl<P, C> NodeOld<P, C>
	where
		P: RadixConf,
		C: Children<Self, Radix = P>,
{
	fn prefix_node(&self, key: &[u8]) -> (&Self, Descent<P>) {
		unimplemented!()
	}
	fn prefix_node_mut(&mut self, key: &[u8]) -> (&mut Self, Descent<P>) {
		unimplemented!()
	}
}

/// Stack of Node to reach a position.
struct NodeStack<'a, N: Node> {
	// TODO use smallvec instead
	stack: Vec<(PositionFor<N>, &'a N)>,
	// The key used with the stack.
	// key: Vec<u8>,
}

// TODO put pointers in node stack.
impl<'a, N: Node> NodeStack<'a, N> {
	fn new() -> Self {
		NodeStack {
			stack: Vec::new(),
		}
	}
}
impl<'a, N: Node> NodeStack<'a, N> {
	fn descend(&self, key: &[u8], dest_position: PositionFor<N>) -> Descent<N::Radix> {
		if let Some(top) = self.stack.last() {
			top.1.descend(key, top.0, dest_position)
		} else {
			// using a random key index for root element
			Descent::Child(PositionFor::<N>::zero(), KeyIndexFor::<N>::zero())
		}
	}
}

pub struct SeekIter<'a, N: Node> {
	trie: &'a Trie<N>,
	dest: &'a [u8],
	dest_position: PositionFor<N>,
	// TODO seekiter could be lighter and not stack, 
	// just keep latest: a stack trait could be use.
	stack: NodeStack<'a, N>,
	reach_dest: bool,
	next: Descent<N::Radix>,
}
pub struct SeekValueIter<'a, N: Node>(SeekIter<'a, N>);
	
impl<N: Node> Trie<N> {
	pub fn seek_iter<'a>(&'a self, key: &'a [u8]) -> SeekIter<'a, N> {
		let dest_position = Position {
			index: key.len(),
			mask: MaskFor::<N::Radix>::empty(),
		};
		self.seek_iter_at(key, dest_position)
	}
	/// Seek non byte aligned nodes.
	pub fn seek_iter_at<'a>(&'a self, key: &'a [u8], dest_position: PositionFor<N>) -> SeekIter<'a, N> {
		let stack = NodeStack::new();
		let reach_dest = false;
		let next = stack.descend(key, dest_position);
		SeekIter {
			trie: self,
			dest: key,
			dest_position,
			stack,
			reach_dest,
			next,
		}
	}
}


impl<'a, N: Node> SeekIter<'a, N> {
	pub fn value_iter(self) -> SeekValueIter<'a, N> {
		SeekValueIter(self)
	}
	fn next_node(&mut self) -> Option<(PositionFor<N>, &'a N)> {
		if self.reach_dest {
			return None;
		}
		match self.next {
			Descent::Child(position, index) => {
				if let Some(parent) = self.stack.stack.last() {
					// TODO stack child
					if let Some(child) = parent.1.get_child(index) {
						let position = position.next::<N::Radix>();
						self.stack.stack.push((position, child));
					} else {
						self.reach_dest = true;
						return None;
					}
				} else {
					// empty trie
					//		// TODO put ref in stack.
					if let Some(node) = self.trie.tree.as_ref() {
						let zero = PositionFor::<N>::zero();
						self.stack.stack.push((zero, node));
					} else {
						self.reach_dest = true;
					}
				}
			},
			Descent::Middle(_position, _index) => {
				self.reach_dest = true;
				return None;
			},
			Descent::Match(_position) => {
				self.reach_dest = true;
			},
		}
		if !self.reach_dest {
			self.next = self.stack.descend(&self.dest, self.dest_position);
		}
		self.stack.stack.last().map(|last| (last.0, last.1))
	}
}

impl<'a, N: Node> Iterator for SeekIter<'a, N> {
	type Item = (&'a [u8], PositionFor<N>, &'a N);
	fn next(&mut self) -> Option<Self::Item> {
		self.next_node().map(|(pos, node)| (self.dest, pos, node))
	}
}

impl<'a, N: Node> Iterator for SeekValueIter<'a, N> {
	type Item = (&'a [u8], &'a [u8]);
	fn next(&mut self) -> Option<Self::Item> {
		loop {
			if let Some((key, _pos, node)) = self.0.next() {
				if let Some(v) = node.value() {
					return Some((key, v))
				}
			} else {
				return None;
			}
		}
	}
}

#[cfg(test)]
mod test {
	use crate::*;

	type Node = NodeOld<Radix256RadixConf, Children256Bis>;

	#[test]
	fn empty_are_equals() {
		let t1 = Trie::<Node>::new();
		let t2 = Trie::<Node>::new();
		assert_eq!(t1, t2);
	}

	#[test]
	fn inserts_are_equals() {
		let mut t1 = Trie::<Node>::new();
		let mut t2 = Trie::<Node>::new();
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
