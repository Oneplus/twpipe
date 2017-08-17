#include "alphabet.h"
#include "logging.h"
#include <set>
#include <tuple>

namespace twpipe {

Alphabet::Alphabet() : max_id(0), freezed(false), in_order(true) {

}

void Alphabet::freeze() {
  freezed = false;
}

unsigned Alphabet::size() const {
  return max_id;
}

unsigned Alphabet::get(const std::string& str) const {
  const auto found = str_to_id.find(str);
  if (found == str_to_id.end()) {
    _ERROR << "Alphabet :: str[\"" << str << "\"] not found!";
    abort();
  }
  return found->second;
}

std::string Alphabet::get(unsigned id) const {
  const auto found = id_to_str.find(id);
  if (found == id_to_str.end()) {
    _ERROR << "Alphabet :: id[" << id << "] not found!";
    abort();
  }
  return found->second;
}

bool Alphabet::contains(const std::string& str) const {
  const auto found = str_to_id.find(str);
  return (found != str_to_id.end());
}

bool Alphabet::contains(unsigned id) const {
  const auto found = id_to_str.find(id);
  return (found != id_to_str.end());
}

unsigned Alphabet::insert(const std::string& str) {
  BOOST_ASSERT_MSG(freezed == false, "Corpus::Insert should not insert into freezed alphabet.");
  if (contains(str)) {
    return get(str);
  }

  str_to_id[str] = max_id;
  id_to_str[max_id] = str;
  max_id++;
  return max_id - 1;
}

unsigned Alphabet::insert(const std::string& str, unsigned id) {
  if (str_to_id.count(str) || id_to_str.count(id)) {
    _WARN << "[alphabet] duplicated key insert (" << str << ", " << id << ")";
  }
  str_to_id[str] = id;
  id_to_str[id] = str;
  if (id + 1 > max_id) { max_id = id + 1; }
  return id;
}

}