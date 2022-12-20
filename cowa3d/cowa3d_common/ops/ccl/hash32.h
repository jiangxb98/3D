#pragma once
#include "macros.h"
#include <limits>

namespace hash {

template <int N> struct ubytes
{
};
template <> struct ubytes<1>
{
  using type = uint8_t;
};
template <> struct ubytes<2>
{
  using type = uint16_t;
};
template <> struct ubytes<4>
{
  using type = uint32_t;
};
template <> struct ubytes<8>
{
  using type = uint64_t;
};
template <class T> using uint_t = typename ubytes<sizeof(T)>::type;
template <class T> HOST_DEVICE_INLINE uint_t<T> uint_view(T x)
{
  return *reinterpret_cast<uint_t<T>*>(&x);
}

template <class T> HOST_DEVICE_INLINE uint_t<T>* uint_view_ptr(T* x)
{
  return reinterpret_cast<uint_t<T>*>(x);
}

template <class K> constexpr K empty_v = std::numeric_limits<K>::max();

struct Murmur3Hash4B
{
  template <typename K> size_t HOST_DEVICE_INLINE operator()(K key) const
  {
    uint_t<K> k = uint_view(key);
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return static_cast<size_t>(k);
  }
};

template <class K, class HashFtor = Murmur3Hash4B> struct LinearHashSet
{
 public:
  using key_type = K;
  static constexpr K EMPTY = empty_v<K>;

 protected:
  key_type* key_table_ = nullptr;
  size_t size_;
  HashFtor hash_ftor_;

 public:
  explicit LinearHashSet(key_type* key_table, size_t size) : key_table_(key_table), size_(size), hash_ftor_() {}
  HOST_DEVICE_INLINE size_t size() const { return size_; }
  HOST_DEVICE_INLINE const key_type* keys() const { return key_table_; }
  HOST_DEVICE_INLINE key_type* keys() { return key_table_; }

  HOST_DEVICE_INLINE size_t insert(const K& key)
  {
    size_t slot = hash_ftor_(key) % this->size_;
    uint_t<K> key_u = uint_view(key);
    for (int i = 0; i < this->size_; i++) {
      uint_t<K> prev = atomicCAS(uint_view_ptr(this->key_table_ + slot), uint_view(EMPTY), key_u);
      if (prev == uint_view(EMPTY) || prev == key_u) { return slot; }
      slot = (slot + 1) % this->size_;
    }
    return empty_v<size_t>;
  }

  HOST_DEVICE_INLINE bool lookup(const K& key) const
  {
    size_t slot = hash_ftor_(key) % this->size_;
    for (int i = 0; i < this->size_; i++) {
      K found = this->key_table_[slot];
      if (found == key) { return true; }
      if (found == EMPTY) { return false; }
      slot = (slot + 1) % this->size_;
    }
    return false;
  }
};

template <class K, class V, class HashFtor = Murmur3Hash4B> struct LinearHashMap : public LinearHashSet<K, HashFtor>
{
 public:
  using Base = LinearHashSet<K, HashFtor>;
  using key_type = typename Base::key_type;
  using value_type = V;
  static constexpr K EMPTY = Base::EMPTY;

 protected:
  value_type* value_table_ = nullptr;

 public:
  explicit LinearHashMap(key_type* key_table, value_type* value_table, size_t size)
      : LinearHashSet<K, HashFtor>(key_table, size), value_table_(value_table)
  {
  }

  HOST_DEVICE_INLINE const value_type* data() const { return value_table_; }
  HOST_DEVICE_INLINE value_type* data() { return value_table_; }

  HOST_DEVICE_INLINE size_t insert_if_empty(const K& key, const V& value)
  {
    size_t slot = hash_ftor_(key) % this->size_;
    uint_t<K> key_u = uint_view(key);
    for (int i = 0; i < this->size_; i++) {
      uint_t<K> prev = atomicCAS(uint_view_ptr(this->key_table_ + slot), uint_view(Base::EMPTY), key_u);
      if (prev == uint_view(Base::EMPTY)) {
        value_table_[slot] = value;
        return slot;
      }
      if (prev == key_u) {
        return slot;
      }
      slot = (slot + 1) % this->size_;
    }
    return empty_v<size_t>;
  }

  HOST_DEVICE_INLINE size_t insert(const K& key, const V& value)
  {
    size_t slot = hash_ftor_(key) % this->size_;
    uint_t<K> key_u = uint_view(key);
    for (int i = 0; i < this->size_; i++) {
      uint_t<K> prev = atomicCAS(uint_view_ptr(this->key_table_ + slot), uint_view(Base::EMPTY), key_u);
      if (prev == uint_view(Base::EMPTY) || prev == key_u) {
        value_table_[slot] = value;
        return slot;
      }
      slot = (slot + 1) % this->size_;
    }
    return empty_v<size_t>;
  }

  HOST_DEVICE_INLINE bool lookup(const K& key, V& value) const
  {
    size_t slot = hash_ftor_(key) % this->size_;
    for (int i = 0; i < this->size_; i++) {
      K found = this->key_table_[slot];
      if (found == key) {
        value = value_table_[slot];
        return true;
      }
      if (found == EMPTY) { return false; }
      slot = (slot + 1) % this->size_;
    }
    return false;
  }
};

}  // namespace hash
