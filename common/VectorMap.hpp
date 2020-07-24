#ifndef VECTOR_MAP_HPP
#define VECTOR_MAP_HPP

#include <vector>
#include <map>
#include <iterator>
#include <cstdint>
#include <cassert>

//#define VM_USE_SPARSEPP // slightly faster but uses more memory, ironically
#ifdef VM_USE_SPARSEPP
#include "sparsepp.h"
#undef VM_TARGET_BUCKET_SIZE
#define VM_TARGET_BUCKET_SIZE (1024*1024)
#undef VM_MAX_NUM_BUCKETS
#define VM_MAX_NUM_BUCKETS 1
#endif

#ifndef VM_TARGET_BUCKET_SIZE
#define VM_TARGET_BUCKET_SIZE 512
#endif

#ifndef VM_MIN_NUM_BUCKETS
#define VM_MIN_NUM_BUCKETS 1
#endif

#ifndef VM_MAX_NUM_BUCKETS
#define VM_MAX_NUM_BUCKETS 16384
#endif

/*temporary definition for debugging*/
//#ifndef __INTEL_COMPILER
//#define __INTEL_COMPILER
//#endif

#ifdef __INTEL_COMPILER
#define NOEXCEPT /* noexcept */
#else
#define NOEXCEPT noexcept
#endif

#include "../common/hash_funcs.h"
template <class Key>
class _hash
{
public:
    size_t operator()(const Key &key) const {
        return MurmurHash3_x64_64(&key, sizeof(Key));
    }
};

using namespace std;
template < class Key, class T, class Hash=_hash<Key>, class Compare=less<Key>, class Equal=equal_to<Key>, class Allocator=allocator< pair< const Key, T> > >
class VectorMap
{

public:
#ifdef VM_USE_SPARSEPP
	typedef spp::sparse_hash_map< Key, T, Hash, Equal, Allocator > Map;
#else
	typedef map<Key, T, Compare, Allocator> Map;
#endif
	typedef vector< Map > VM;
	typedef Key key_type; // the first template parameter (Key)	
	typedef T mapped_type; // the second template parameter (T)	
	typedef pair< const key_type, mapped_type > value_type; // pair<const key_type, mapped_type>	
	typedef Hash hasher; // the third template parameter (Hash)	defaults to: hash<key_type>
	typedef Compare comparer; // the fourth template parameter(Compare) defaults to: less<Key>
	typedef Equal key_equal; // the fifth template parameter (Pred)	defaults to: equal_to<key_type>
	typedef Allocator allocator_type; // the sixthk template parameter (Alloc)	defaults to: allocator<value_type>
	typedef typename Allocator::reference reference;
	typedef typename Allocator::const_reference const_reference;
	typedef typename Allocator::pointer pointer; // Alloc::pointer	for the default allocator: value_type*
	typedef typename Allocator::const_pointer const_pointer; // Alloc::const_pointer	for the default allocator: const value_type*
	typedef uint64_t size_type; //	an unsigned integral type	usually the same as size_t
	typedef int64_t difference_type; //	a signed integral type	usually the same as ptrdiff_t

	static size_type nextPowerOf2(size_type minimumSize) {
		size_type power = 1;
		while(power < minimumSize)
		    power<<=1;
		if (power < VM_MIN_NUM_BUCKETS) power = VM_MIN_NUM_BUCKETS;
		if (power > VM_MAX_NUM_BUCKETS) power = VM_MAX_NUM_BUCKETS;
		return power;
	}
	static size_type &targetBucketSize() {
		static size_type _ = VM_TARGET_BUCKET_SIZE;
		return _;
	}

private:
	VM _data;
	size_type _vectorMask;
	hasher _hasher;

public:
	VectorMap(size_t reserveSize = VM_MIN_NUM_BUCKETS * VM_TARGET_BUCKET_SIZE): _data(), _vectorMask(0), _hasher() {
		size_t numBuckets = nextPowerOf2( (reserveSize+targetBucketSize()-1) / targetBucketSize());
		_data.resize(numBuckets);
		_vectorMask = numBuckets-1;
//fprintf(stderr, "Created VectorMap %p with %lld buckets for %lld reservation\n", this, numBuckets, reserveSize);
	}
	VectorMap(size_t reserveSize, size_t numBuckets): _data(), _vectorMask(0), _hasher() {
		numBuckets = nextPowerOf2(numBuckets);
		_data.resize(numBuckets);
		_vectorMask = numBuckets-1;
//fprintf(stderr, "Created VectorMap %p with %lld buckets for %lld reservation\n", this, numBuckets, reserveSize);
	}
	~VectorMap() {}
	bool empty() const NOEXCEPT { return size() == 0; }
	size_type size() const NOEXCEPT {
		size_type s = 0, max = 0, min = 0;
		typename VM::const_iterator it;
		for (it = _data.begin(); it != _data.end(); it++) {
			size_type s1 = it->size();
			if (max < s1) max = s1;
			if (min == 0 || min > s1) min = s1;
			s += it->size();
		}
//fprintf(stderr, "max: %lld, min: %lld, total: %lld\n", (long long int) max, (long long int) min, (long long int) s);
		return s;
	}
	class iterator {
	public:
		VM *_vm;
		size_type _b;
		typename Map::iterator _m;
	public:
		iterator(VM &vm, size_type b, typename Map::iterator m): _vm(&vm), _b(b), _m(m) {
                    prime();
                }
		iterator(const iterator &copy): _vm(copy._vm), _b(copy._b), _m(copy._m) {}
		~iterator() {}
		void prime() {
			while (_b < _vm->size() && _m == (*_vm)[_b].end()) increment();
		}
		void increment() {
			if (_b >= _vm->size()) { fprintf(stderr, "ERROR!! increment on iterator beyond end\n"); return; }
			do {
				if (_m == (*_vm)[_b].end()) { 
					_b++;
					if (_b == _vm->size()) { assert(_m == _vm->back().end()); break; }
					_m = (*_vm)[_b].begin();
				} else {
					_m++;
				}
			} while (_m == (*_vm)[_b].end());
		}
		value_type &operator*() { return *_m; }
		value_type *operator->() { return &(*_m); }
		iterator &operator++() { increment(); return *this; }
		iterator operator++(int) { iterator old(*this); increment(); return old; }
		bool operator==(const iterator &other) const { return (_vm == other._vm && _b == other._b && _m == other._m); }
		bool operator!=(const iterator &other) const { return !(*this == other); }
	};
	class const_iterator {
	public:
		const VM *_vm;
		size_type _b;
		typename Map::const_iterator _m;
	public:
		const_iterator(const VM &vm, size_type b, typename Map::const_iterator m): _vm(&vm), _b(b), _m(m) {
                    prime();
                }
		const_iterator(const const_iterator &copy): _vm(copy._vm), _b(copy._b), _m(copy._m) {}
		const_iterator(const iterator &copy): _vm(copy._vm), _b(copy._b), _m(copy._m) {}
		~const_iterator() {}
		void prime() {
			while (_b < _vm->size() && _m == (*_vm)[_b].end()) increment();
		}
		void increment() {
			if (_b >= _vm->size()) { fprintf(stderr, "ERROR!! increment on const_iterator beyond end\n"); return; }
			do {
				if (_m == (*_vm)[_b].end()) { 
					_b++;
					if (_b == _vm->size()) { assert(_m == (*_vm).back().end()); break; }
					_m = (*_vm)[_b].begin();
				} else {
					_m++;
				}
			} while (_m == (*_vm)[_b].end());
		}
		const value_type &operator*() { return *_m; }
		const value_type *operator->() { return &(*_m); }
		const_iterator &operator++() { increment(); return *this; }
		const_iterator operator++(int) { const_iterator old(*this); increment(); return old; }
		bool operator==(const const_iterator &other) const { return (_vm == other._vm && _b == other._b && _m == other._m); }
		bool operator!=(const const_iterator &other) const { return !(*this == other); }
	};
	iterator begin() NOEXCEPT { assert(!_data.empty()); return iterator(_data, 0, _data[0].begin()); }
	const_iterator begin() const NOEXCEPT { assert(!_data.empty()); return const_iterator(_data, 0, _data[0].begin()); }
	iterator end() NOEXCEPT { assert(!_data.empty()); return iterator(_data, _data.size(), _data.back().end()); }
	const_iterator end() const NOEXCEPT { assert(!_data.empty()); return const_iterator(_data, _data.size(), _data.back().end()); }

	
	mapped_type& operator[] ( const key_type& k ) { return this->at(k); }
	mapped_type& operator[] ( key_type&& k ) { return this->at(k); }
	mapped_type& at ( const key_type& k ) { return getBucket(k).at(k); }
	const mapped_type& at ( const key_type& k ) const { return getBucket(k).at(k); }
	iterator find ( const key_type& k ) {
		size_type bidx = getBucketIdx(k);
		typename Map::iterator it = _data[bidx].find(k);
		if (it == _data[bidx].end())
			return end();
		else
			return iterator(_data, bidx, it);
	}
	const_iterator find ( const key_type& k ) const {
		size_type bidx = getBucketIdx(k);
		typename Map::const_iterator it = _data[bidx].find(k);
		if (it == _data[bidx].end())
			return end();
		else
			return const_iterator(_data, bidx, it);
	}

	pair<iterator,bool> insert ( const value_type& val ) {
		size_type bidx = getBucketIdx(val.first);
		pair<typename Map::iterator, bool> ret = _data[bidx].insert(val);
		pair<iterator, bool> ret2( iterator(_data, bidx, ret.first), ret.second );
		return ret2;
	}
	iterator insert ( const_iterator hint, const value_type& val ) {
		// ignore the hint
		return insert(val).first;
	}
	template <class InputIterator> void insert ( InputIterator first, InputIterator last ) {
		while(first != last) {
			insert(*first);
			first++;
		}
	}

	iterator erase ( const_iterator position ) {
		size_type bidx = getBucketIdx( position->first );
		typename Map::const_iterator it = position._m;
		assert( it != _data[ bidx ].end() );
#ifdef __INTEL_COMPILER
		typename Map::key_type key = it->first;
		typename Map::iterator it2 = _data[bidx].find(key);
		it2++; // This should remain valid
		_data[ bidx ].erase( key );
		return iterator(_data, bidx, it2);
#else
		typename Map::iterator ret = _data[ bidx ].erase( it );
		return iterator(_data, bidx, ret);
#endif
	}
	size_type erase ( const key_type& k ) {
		size_type bidx = getBucketIdx( k );
		return _data[bidx].erase(k);
	}
	iterator erase ( const_iterator first, const_iterator last ) {
		iterator it = end();
		while (first != last) 
			it = erase(first++);
		return it;
	}
	void clear() NOEXCEPT {
		for(typename VM::iterator it = _data.begin(); it != _data.end() ; it++)
			it->clear();
	}
	void swap ( VectorMap& other ) {
		_data.swap(other._data);
		::swap(_vectorMask, other._vectorMask);
		::swap(_hasher, other._hasher);
	}
		
	void reserve ( size_type num_elements ) {
		size_type bucketSize = targetBucketSize();
		size_type power = nextPowerOf2( (num_elements + bucketSize - 1) / bucketSize );
		rehash(power);
		assert(_data.size() == power);
		size_type numPerBucket = num_elements * 15 / power / 10; // 150%
#ifdef VM_USE_SPARSEPP
//fprintf(stderr, "VectorMap (Sparsepp): reserving %lld entries for %lld buckets\n", (long long int) numPerBucket, (long long int) power);
		for(size_type i = 0; i < power; i++) {
			_data[i].reserve(numPerBucket);
		}
#else
//fprintf(stderr, "VectorMap (std::map): expecting %lld entries for %lld buckets\n", (long long int) numPerBucket, (long long int) power);
#endif
	}
	void rehash( size_type n ) {
		if ((n & (n-1)) != 0x0)
			n = nextPowerOf2( n ); // n must be a power of 2
		assert((n & (n-1)) == 0x0);
		size_type mySize = size();
		if (_data.size() != n && mySize > 0) {
			VectorMap newvm(mySize, n);
			for(const_iterator it = begin(); it != end(); it++) {
				newvm.insert(*it);
			}
			assert( newvm.size() == mySize );
			this->swap(newvm);
		} else if (mySize == 0) {
			_data.resize(n);
			_vectorMask = n-1;
		}
		assert(_data.size() == n);
		assert(_vectorMask == n-1);
//fprintf(stderr, "rehashed VectorMap %p for %lld buckets for %lld reservation\n", this, n, mySize);
	}
	size_type getBucketIdx(const key_type &k) const {
		assert(!_data.empty());
		assert( _vectorMask + 1 == _data.size() );
		if (_data.size() == 1) return 0;
		size_type h = _hasher(k);
		h >>= 32; // use the high bits as the underlying Map will likely use the low ones...
		return (h & _vectorMask);
	}
	Map &getBucket(const key_type &k) {
		return _data[ getBucketIdx(k) ];
	}
	const Map &getBucket(const key_type &k) const {
		return _data[ getBucketIdx(k) ];
	}
};

#endif
