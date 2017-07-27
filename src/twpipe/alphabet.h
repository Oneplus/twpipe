#ifndef __TWPIPE_ALPHABET_H__
#define __TWPIPE_ALPHABET_H__

#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>

namespace twpipe {

struct Alphabet {
  typedef std::unordered_map<std::string, unsigned> StringToIdMap;
  typedef std::unordered_map<unsigned, std::string> IdToStringMap;

  unsigned max_id;
  StringToIdMap str_to_id;
  IdToStringMap id_to_str;
  bool freezed;
  bool in_order;

  Alphabet();

  void freeze();
  unsigned size() const;
  unsigned get(const std::string& str) const;
  std::string get(unsigned id) const;
  bool contains(const std::string& str) const;
  bool contains(unsigned id) const;
  unsigned insert(const std::string& str);
  unsigned insert(const std::string& str, unsigned id);
};

}

struct HashVector : public std::vector<unsigned> {
  bool operator == (const HashVector& other) const {
    if (size() != other.size()) { return false; }
    for (unsigned i = 0; i < size(); ++i) {
      if (at(i) != other.at(i)) { return false; }
    }
    return true;
  }
};

namespace std {
template<>
struct hash<HashVector> {
  std::size_t operator()(const HashVector& values) const {
    size_t seed = 0;
    boost::hash_range(seed, values.begin(), values.end());
    return seed;
  }
};
}

#endif  //  end for __TWPIPE_ALPHABET_H__