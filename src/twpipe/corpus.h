#ifndef __TWPIPE_CORPUS_H__
#define __TWPIPE_CORPUS_H__

#include <unordered_map>
#include <vector>
#include <set>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

enum EmbeddingType {kStaticEmbeddings, kContextualEmbeddings};

struct InputUnit {
  std::vector<unsigned> cids;  // list of character ID
  unsigned wid;     // form ID
  unsigned pid;     // postag ID
  unsigned aux_wid; // copy of form ID
  std::string word;
  std::string lemma;
  std::string postag;
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

unsigned utf8_len(unsigned char x);
char32_t utf8_to_unicode_first_(const std::string & s);

struct Corpus {
  const static char* UNK;
  const static char* BAD0;
  const static char* SPACE;
  const static char* ROOT;
  const static unsigned BAD_HED;
  const static unsigned BAD_DEL;

  unsigned n_train;
  unsigned n_devel;

  std::unordered_map<unsigned, Instance> training_data;
  std::unordered_map<unsigned, Instance> devel_data;

  std::set<unsigned> training_vocab;
  std::unordered_map<unsigned, unsigned> counter;

  Corpus();

  static void vector_to_input_units(const std::vector<std::string> & words,
                                    const std::vector<std::string> & postags,
                                    InputUnits & units);

  static void vector_to_parse_units(const std::vector<unsigned>& heads,
                                    const std::vector<unsigned>& deprels,
                                    ParseUnits& parse,
                                    bool has_pseudo_root=true);

  static void parse_units_to_vector(const ParseUnits& parse,
                                    std::vector<unsigned>& heads,
                                    std::vector<unsigned>& deprels);

  static void parse_units_to_vector(const ParseUnits & units,
                                    std::vector<unsigned> & heads,
                                    std::vector<std::string> & deprels,
                                    bool add_pseduo_root = false);

  void load_training_data(const std::string& filename);

  void load_devel_data(const std::string& filename);

  void parse_data(const std::string& data,
                  Instance & inst,
                  bool train);

  void get_vocabulary_and_word_count();
};

}

#endif  //  end for CORPUS_H
