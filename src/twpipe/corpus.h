#ifndef __TWPIPE_CORPUS_H__
#define __TWPIPE_CORPUS_H__

#include <unordered_map>
#include <vector>
#include <set>
#include <boost/regex.hpp>
#include "alphabet.h"

namespace twpipe {

struct InputUnit {
  std::vector<unsigned> cids;  // list of character ID
  unsigned wid;     // form ID
  unsigned nid;     // normalized form ID
  unsigned pid;     // postag ID
  unsigned aux_wid; // copy of form ID
  std::string word;
  std::string norm_word;
  std::string lemma;
  std::string feature;
};

struct ParseUnit {
  unsigned head;
  unsigned deprel;
};

typedef std::vector<InputUnit> InputUnits;
typedef std::vector<ParseUnit> ParseUnits;

struct Instance {
  std::string raw_sentence;
  InputUnits input_units;
  ParseUnits parse_units;
};

void parse_to_vector(const ParseUnits& parse,
                     std::vector<unsigned>& heads,
                     std::vector<unsigned>& deprels);

void vector_to_parse(const std::vector<unsigned>& heads,
                     const std::vector<unsigned>& deprels,
                     ParseUnits& parse);

unsigned utf8_len(unsigned char x);

typedef std::unordered_map<unsigned, std::vector<float> > IntEmbeddingType;
typedef std::unordered_map<std::string, std::vector<float> > StrEmbeddingType;

struct Corpus {
  const static char* UNK;
  const static char* BAD0;
  const static char* SPACE;
  const static char* ROOT;
  const static unsigned BAD_HED;
  const static unsigned BAD_DEL;

  unsigned n_train;
  unsigned n_devel;

  Alphabet word_map;  //  alphabet of word
  Alphabet norm_map;  //  alphabet of normalized word
  Alphabet char_map;  //  alphabet of characters
  Alphabet pos_map;   //  alphabet of postag
  Alphabet deprel_map;

  std::unordered_map<unsigned, Instance> training_data;
  std::unordered_map<unsigned, Instance> devel_data;

  std::set<unsigned> training_vocab;
  std::unordered_map<unsigned, unsigned> counter;

  boost::regex url_regex;
  boost::regex user_regex;
  boost::regex smile_regex;
  boost::regex lolface_regex;
  boost::regex sadface_regex;
  boost::regex neuralface_regex;
  boost::regex heart_regex;
  boost::regex number_regex;
  boost::regex repeat_regex;
  boost::regex elong_regex;

  Corpus();

  // dealing with username, url, emoticon, expressive lengthening
  // match with the glove normalization process.
  std::string normalize(const std::string & word) const;

  void load_training_data(const std::string& filename);

  void load_devel_data(const std::string& filename);

  void parse_data(const std::string& data,
                  Instance & inst,
                  bool train);

  void get_vocabulary_and_word_count();

  unsigned get_or_add_word(const std::string& word);

  void stat();
};

void load_word_embeddings(const std::string& embedding_file,
                          unsigned pretrained_dim,
                          IntEmbeddingType & pretrained,
                          Alphabet & norm_map);

void load_empty_embeddings(unsigned pretrained_dim,
                           IntEmbeddingType & pretrained,
                           Alphabet & norm_map);


void load_word_embeddings(const std::string& embedding_file,
                          unsigned pretrained_dim,
                          StrEmbeddingType & pretrained);

void load_empty_embeddings(unsigned pretrained_dim,
                           IntEmbeddingType & pretrained);

}

#endif  //  end for CORPUS_H
