#include "corpus.h"
#include "alphabet_collection.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

namespace twpipe {

const char* Corpus::UNK = "_UNK_";
const char* Corpus::BAD0 = "_BAD0_";
const char* Corpus::ROOT = "_ROOT_";
const char* Corpus::SPACE = " ";
const unsigned Corpus::BAD_HED = 10000;
const unsigned Corpus::BAD_DEL = 10000;

void Corpus::parse_units_to_vector(const ParseUnits& parse,
                                   std::vector<unsigned>& heads,
                                   std::vector<unsigned>& deprels) {
  heads.clear();
  deprels.clear();
  /// The first unit is the pseudo root.
  for (unsigned i = 0; i < parse.size(); ++i) {
    heads.push_back(parse[i].head);
    deprels.push_back(parse[i].deprel);
  }
}

void Corpus::vector_to_parse_units(const std::vector<unsigned>& heads,
                                   const std::vector<unsigned>& deprels,
                                   ParseUnits& parse,
                                   bool has_pseudo_root) {
  parse.clear();
  BOOST_ASSERT_MSG(heads.size() == deprels.size(),
                   "In corpus.cc: vector_to_parse, #heads should be equal to #deprels");
    
  ParseUnit parse_unit;
  if (!has_pseudo_root) {
    parse_unit.head = Corpus::BAD_HED;
    parse_unit.deprel = Corpus::BAD_DEL;
    parse.push_back(parse_unit);
  }  
  for (unsigned i = 0; i < heads.size(); ++i) {
    parse_unit.head = heads[i];
    parse_unit.deprel = deprels[i];
    parse.push_back(parse_unit);
  }
}

void Corpus::vector_to_input_units(const std::vector<std::string>& words,
                                   const std::vector<std::string>& postags,
                                   InputUnits & units) {
  // The first element is the pseudo root.
  units.clear();

  Alphabet & word_map = AlphabetCollection::get()->word_map;
  Alphabet & char_map = AlphabetCollection::get()->char_map;
  Alphabet & pos_map = AlphabetCollection::get()->pos_map;

  InputUnit unit;
  unit.wid = word_map.get(Corpus::ROOT);
  unit.pid = pos_map.get(Corpus::ROOT);
  unit.aux_wid = unit.wid;
  unit.word = Corpus::ROOT;
  unit.postag = Corpus::ROOT;
  unit.lemma = Corpus::ROOT;
  unit.feature = Corpus::ROOT;
  units.push_back(unit);

  for (unsigned i = 0; i < words.size(); ++i) {
    const std::string & word = words[i];
    const std::string & postag = postags[i];

    unit.wid = (word_map.contains(word) ? word_map.get(word) : word_map.get(Corpus::UNK));
    unit.pid = pos_map.get(postag);
    unit.aux_wid = unit.wid;
    unit.word = word;
    unit.postag = postag;

    unsigned cur = 0;
    unit.cids.clear();
    while (cur < word.size()) {
      unsigned len = utf8_len(word[cur]);
      std::string ch_str = word.substr(cur, len);
      unit.cids.push_back(
        char_map.contains(ch_str) ? char_map.get(ch_str) : char_map.get(Corpus::UNK)
      );
      cur += len;
    }
    units.push_back(unit);
  }
}

void Corpus::parse_units_to_vector(const ParseUnits & units,
                                   std::vector<unsigned>& heads,
                                   std::vector<std::string>& deprels,
                                   bool add_pseudo_root) {
  heads.clear();
  deprels.clear();

  if (add_pseudo_root) {
    heads.push_back(Corpus::BAD_HED);
    deprels.push_back(Corpus::ROOT);
  }
  // The first element in units is pseduo root.
  for (unsigned i = 1; i < units.size(); ++i) {
    heads.push_back(units[i].head);
    deprels.push_back(AlphabetCollection::get()->deprel_map.get(units[i].deprel));
  }
}

Corpus::Corpus() :
  n_train(0),
  n_devel(0) {
}

void Corpus::load_training_data(const std::string& filename) {
  _INFO << "[corpus] reading training data from: " << filename;

  Alphabet & word_map = AlphabetCollection::get()->word_map;
  Alphabet & char_map = AlphabetCollection::get()->char_map;
  Alphabet & pos_map = AlphabetCollection::get()->pos_map;

  word_map.insert(Corpus::BAD0);
  word_map.insert(Corpus::UNK);
  word_map.insert(Corpus::ROOT);

  char_map.insert(Corpus::BAD0);
  char_map.insert(Corpus::UNK);
  char_map.insert(Corpus::ROOT);
  char_map.insert(Corpus::SPACE);

  pos_map.insert(Corpus::ROOT);

  std::ifstream in(filename);
  if (!in) {
    _ERROR << "[corpus] failed to open the training file.";
    exit(1);
  }

  n_train = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      // end for an instance.
      parse_data(data, training_data[n_train], true);
      data = "";
      ++n_train;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, training_data[n_train], true);
    ++n_train;
  }

  _INFO << "[corpus] loaded " << n_train << " training sentences.";
}

void Corpus::load_devel_data(const std::string& filename) {
  _INFO << "[corpus] reading development data from: " << filename;

  Alphabet & word_map = AlphabetCollection::get()->word_map;

  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "[corpus] BAD0 and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  if (!in) {
    _ERROR << "[corpus] failed to open the devel file.";
    exit(1);
  }

  n_devel = 0;
  std::string data = "";
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      parse_data(data, devel_data[n_devel], false);
      data = "";
      ++n_devel;
    } else {
      data += (line + "\n");
    }
  }
  if (data.size() > 0) {
    parse_data(data, devel_data[n_devel], false);
    ++n_devel;
  }

  _INFO << "[corpus] loaded " << n_devel << " development sentences.";
}

unsigned utf8_len(unsigned char x) {
  if (0 == (0x80 & x))         { return 1; }
  else if (0xc0 == (0xe0 & x)) { return 2; }
  else if (0xe0 == (0xf0 & x)) { return 3; }
  else if (0xf0 == (0xf8 & x)) { return 4; }
  else if (0xf8 == (0xfc & x)) { return 5; }
  else if (0xfc == (0xfe & x)) { return 6; }
  assert(false);
}

char32_t utf8_to_unicode_first_(const std::string & s) {
  char32_t wc = 0;
  unsigned char c = s[0];
  if ((c & 0x80) == 0) { wc = c; }
  else if ((c & 0xE0) == 0xC0) { wc = (s[0] & 0x1F) << 6; wc |= (s[1] & 0x3F); }
  else if ((c & 0xF0) == 0xE0) { wc = (s[0] & 0xF) << 12; wc |= (s[1] & 0x3F) << 6;  wc |= (s[2] & 0x3F); }
  else if ((c & 0xF8) == 0xF0) { wc = (s[0] & 0x7) << 18; wc |= (s[1] & 0x3F) << 12; wc |= (s[2] & 0x3F) << 6;  wc |= (s[3] & 0x3F); }
  else if ((c & 0xFC) == 0xF8) { wc = (s[0] & 0x3) << 24; wc |= (s[0] & 0x3F) << 18; wc |= (s[0] & 0x3F) << 12; wc |= (s[0] & 0x3F) << 6;  wc |= (s[0] & 0x3F); }
  else if ((c & 0xFE) == 0xFC) { wc = (s[0] & 0x1) << 30; wc |= (s[0] & 0x3F) << 24; wc |= (s[0] & 0x3F) << 18; wc |= (s[0] & 0x3F) << 12; wc |= (s[0] & 0x3F) << 6; wc |= (s[0] & 0x3F); }
  else { assert(false); }
  return wc;
}

// id form lemma cpos pos feat head deprel phead pdeprel
// 0  1    2     3    4   5    6     7     8     9
void Corpus::parse_data(const std::string& data, Instance & inst, bool train) {
  std::stringstream S(data);
  std::string line;

  inst.input_units.clear();
  inst.parse_units.clear();

  InputUnit input_unit;
  ParseUnit parse_unit;

  Alphabet & word_map = AlphabetCollection::get()->word_map;
  Alphabet & char_map = AlphabetCollection::get()->char_map;
  Alphabet & pos_map = AlphabetCollection::get()->pos_map;
  Alphabet & deprel_map = AlphabetCollection::get()->deprel_map;

  // dummy root at first.
  input_unit.wid = word_map.get(ROOT);
  input_unit.pid = pos_map.get(ROOT);
  input_unit.aux_wid = input_unit.wid;
  input_unit.word = ROOT;
  input_unit.postag = ROOT;
  input_unit.lemma = ROOT;
  input_unit.feature = ROOT;
  inst.input_units.push_back(input_unit);

  parse_unit.head = BAD_HED;
  parse_unit.deprel = BAD_DEL;
  inst.parse_units.push_back(parse_unit);

  std::string guessed_raw_sentence = "";
  while (std::getline(S, line)) {
    std::vector<std::string> tokens;
    boost::algorithm::trim(line);
    if (boost::algorithm::starts_with(line, "# text = ")) {
      inst.raw_sentence = line.substr(9);
    } else if (boost::algorithm::starts_with(line, "#")) {
      continue;
    } else {
      boost::algorithm::split(tokens, line, boost::is_any_of("\t"), boost::token_compress_on);
      BOOST_ASSERT_MSG(tokens.size() > 6, "[corpus] illegal conllu format, number of column less than 6.");      
      
      if (train) {
        const std::string & word = input_unit.word = tokens[1];
        const std::string & postag = input_unit.postag = tokens[3];
        input_unit.lemma = tokens[2];
        input_unit.feature = tokens[5];

        input_unit.wid = word_map.insert(word);
        input_unit.pid = pos_map.insert(postag);
        input_unit.aux_wid = input_unit.wid;

        unsigned cur = 0;
        input_unit.cids.clear();
        while (cur < word.size()) {
          unsigned len = utf8_len(word[cur]);
          input_unit.cids.push_back(char_map.insert(word.substr(cur, len)));
          cur += len;
        }
        inst.input_units.push_back(input_unit);

        parse_unit.head = boost::lexical_cast<unsigned>(tokens[6]);
        parse_unit.deprel = deprel_map.insert(tokens[7]);
        inst.parse_units.push_back(parse_unit);
      } else {
        const std::string & word = input_unit.word = tokens[1];
        const std::string & postag = input_unit.postag = tokens[3];
        input_unit.lemma = tokens[2];
        input_unit.feature = tokens[5];

        input_unit.wid = (word_map.contains(word) ? word_map.get(word) : word_map.get(UNK));
        input_unit.pid = pos_map.get(postag);
        input_unit.aux_wid = input_unit.wid;

        unsigned cur = 0;
        input_unit.cids.clear();
        while (cur < word.size()) {
          unsigned len = utf8_len(word[cur]);
          std::string ch_str = word.substr(cur, len);
          input_unit.cids.push_back(
            char_map.contains(ch_str) ? char_map.get(ch_str) : char_map.get(Corpus::UNK)
          );
          cur += len;
        }
        inst.input_units.push_back(input_unit);

        parse_unit.head = boost::lexical_cast<unsigned>(tokens[6]);
        parse_unit.deprel = deprel_map.insert(tokens[7]);
        inst.parse_units.push_back(parse_unit);
      }
      if (tokens[9] == "SpaceAfter=No" || tokens[9] == "SpaceAfter=\\n") {
        guessed_raw_sentence += tokens[1];
      } else {
        guessed_raw_sentence += (tokens[1] + " ");
      }
    }
  }
  if (inst.raw_sentence == "") {
    inst.raw_sentence = boost::algorithm::trim_copy(guessed_raw_sentence);
  }
}

void Corpus::get_vocabulary_and_word_count() {
  for (auto& payload : training_data) {
    for (auto& item : payload.second.input_units) {
      training_vocab.insert(item.wid);
      ++counter[item.wid];
    }
  }
}


}