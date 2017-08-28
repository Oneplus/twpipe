#ifndef __TWPIPE_CHAR_WCLUSTER_POSTAG_MODEL_H__
#define __TWPIPE_CHAR_WCLUSTER_POSTAG_MODEL_H__

#include "postag_model.h"
#include "twpipe/layer.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/corpus.h"
#include "twpipe/cluster.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"

namespace twpipe {

template <class RNNBuilderType>
struct CharacterRNNWithClusterPostagModel : public PostagModel {
  const static char* name;
  BiRNNLayer<RNNBuilderType> char_rnn;
  BiRNNLayer<RNNBuilderType> word_rnn;
  RNNLayer<RNNBuilderType> cluster_rnn;
  SymbolEmbedding char_embed;
  SymbolEmbedding pos_embed;
  SymbolEmbedding cluster_embed;
  InputLayer embed_input;
  Merge3Layer merge;
  DenseLayer dense;
  dynet::Parameter p_unk_cluster;
  dynet::Expression unk_cluster;

  unsigned char_size;
  unsigned char_dim;
  unsigned char_hidden_dim;
  unsigned char_n_layers;
  unsigned word_dim;
  unsigned word_hidden_dim;
  unsigned word_n_layers;
  unsigned cluster_dim;
  unsigned cluster_hidden_dim;
  unsigned cluster_n_layers;
  unsigned pos_dim;
  unsigned root_pos_id;

  CharacterRNNWithClusterPostagModel(dynet::ParameterCollection & model,
                                     unsigned char_size,
                                     unsigned char_dim,
                                     unsigned char_hidden_dim,
                                     unsigned char_n_layers,
                                     unsigned embed_dim,
                                     unsigned word_hidden_dim,
                                     unsigned word_n_layers,
                                     unsigned cluster_dim,
                                     unsigned cluster_hidden_dim,
                                     unsigned cluster_n_layers,
                                     unsigned pos_dim) :
    PostagModel(model),
    char_rnn(model, char_n_layers, char_dim, char_hidden_dim, false),
    cluster_rnn(model, cluster_n_layers, cluster_dim, cluster_hidden_dim, false),
    word_rnn(model, word_n_layers, char_hidden_dim + char_hidden_dim + cluster_hidden_dim + embed_dim, word_hidden_dim),
    char_embed(model, char_size, char_dim),
    pos_embed(model, AlphabetCollection::get()->pos_map.size(), pos_dim),
    cluster_embed(model, 2, cluster_dim),
    embed_input(embed_dim),
    merge(model, word_hidden_dim, word_hidden_dim, pos_dim, word_hidden_dim),
    dense(model, word_hidden_dim, AlphabetCollection::get()->pos_map.size()),
    char_size(char_size),
    char_dim(char_dim),
    char_hidden_dim(char_hidden_dim),
    char_n_layers(char_n_layers),
    word_hidden_dim(word_hidden_dim),
    word_n_layers(word_n_layers),
    cluster_dim(cluster_dim),
    cluster_hidden_dim(cluster_hidden_dim),
    cluster_n_layers(cluster_n_layers),
    p_unk_cluster(model.add_parameters({cluster_hidden_dim})),
    pos_dim(pos_dim) {
    _INFO << "[postag|model] name = " << name;
    _INFO << "[postag|model] number of character types = " << char_size;
    _INFO << "[postag|model] character dimension = " << char_dim;
    _INFO << "[postag|model] character rnn hidden dimension = " << char_hidden_dim;
    _INFO << "[postag|model] character rnn number layers = " << char_n_layers;
    _INFO << "[postag|model] pre-trained word embedding dimension = " << embed_dim;
    _INFO << "[postag|model] cluster bit dimension = " << cluster_dim;
    _INFO << "[postag|model] cluster rnn hidden dimension = " << cluster_hidden_dim;
    _INFO << "[postag|model] cluster rnn number layers = " << cluster_n_layers;
    _INFO << "[postag|model] word rnn hidden dimension = " << word_hidden_dim;
    _INFO << "[postag|model] word rnn number layers = " << word_n_layers;
    _INFO << "[postag|model] postag hidden dimension = " << pos_dim;

    root_pos_id = AlphabetCollection::get()->pos_map.get(Corpus::ROOT);
  }

  void new_graph(dynet::ComputationGraph & cg) {
    char_rnn.new_graph(cg);
    cluster_rnn.new_graph(cg);
    word_rnn.new_graph(cg);
    char_embed.new_graph(cg);
    cluster_embed.new_graph(cg);
    pos_embed.new_graph(cg);
    embed_input.new_graph(cg);
    merge.new_graph(cg);
    dense.new_graph(cg);
    
    unk_cluster = dynet::parameter(cg, p_unk_cluster);
  }

  void build_input_layer(const std::vector<std::string> & words,
                         std::vector<dynet::Expression> & word_exprs) {
    Alphabet & char_map = AlphabetCollection::get()->char_map;
    Alphabet & pos_map = AlphabetCollection::get()->pos_map;

    std::vector<std::vector<float>> embeddings;
    WordEmbedding::get()->render(words, embeddings);

    std::vector<std::string> clusters;
    WordCluster::get()->render(words, clusters);

    unsigned n_words = words.size();
    word_exprs.resize(n_words);

    for (unsigned i = 0; i < n_words; ++i) {
      std::string word = words[i];
      unsigned len = 0;
      std::vector<unsigned> cids;
      for (unsigned j = 0; j < word.size(); j += len) {
        len = utf8_len(word[j]);
        std::string ch = word.substr(j, len);
        unsigned cid = (char_map.contains(ch) ? char_map.get(ch) : char_map.get(Corpus::UNK));
        cids.push_back(cid);
      }

      unsigned n_chars = cids.size();
      std::vector<dynet::Expression> char_exprs(n_chars);
      for (unsigned j = 0; j < n_chars; ++j) {
        char_exprs[j] = char_embed.embed(cids[j]);
      }
      char_rnn.add_inputs(char_exprs);
      auto payload = char_rnn.get_final();

      const std::string & cluster_type = clusters[i];
      dynet::Expression cluster_expr;
      if (cluster_type == Corpus::UNK) {
        cluster_expr = unk_cluster;
      } else {
        std::vector<dynet::Expression> bits_exprs(cluster_type.size());
        for (unsigned j = 0; j < cluster_type.size(); ++j) {
          bits_exprs[j] = cluster_embed.embed(cluster_type[j] == '0' ? 0 : 1);
        }
        cluster_rnn.add_inputs(bits_exprs);
        cluster_expr = cluster_rnn.get_final();
      }
      word_exprs[i] = dynet::concatenate({ payload.first, payload.second, cluster_expr, embed_input.get_output(embeddings[i]) });
    }
  }

  void decode(const std::vector<std::string> & words, std::vector<std::string> & tags) {
    unsigned n_words = words.size();
    std::vector<dynet::Expression> word_exprs;
    build_input_layer(words, word_exprs);    

    word_rnn.add_inputs(word_exprs);
    tags.resize(n_words);
    unsigned prev_label = root_pos_id;
    for (unsigned i = 0; i < n_words; ++i) {
      auto payload = word_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(
        merge.get_output(payload.first, payload.second, pos_embed.embed(prev_label))
      ));
      std::vector<float> scores = dynet::as_vector((char_embed.cg)->get_value(logits));
      unsigned label = std::max_element(scores.begin(), scores.end()) - scores.begin();

      tags[i] = AlphabetCollection::get()->pos_map.get(label);
      prev_label = label;
    }
  }

  dynet::Expression objective(const Instance & inst) override {
    // embeddings counting w/o pseudo root.
    std::vector<std::string> words;
    for (unsigned i = 1; i < inst.input_units.size(); ++i) {
      words.push_back(inst.input_units[i].word);
    }
   
    const InputUnits & input_units = inst.input_units;
    unsigned n_words_w_root = input_units.size();
    unsigned n_words = n_words_w_root - 1;

    std::vector<dynet::Expression> word_exprs;
    build_input_layer(words, word_exprs);

    std::vector<unsigned> labels(n_words);
    for (unsigned i = 1; i < n_words_w_root; ++i) {
      labels[i - 1] = input_units[i].pid;
    }

    word_rnn.add_inputs(word_exprs);
    std::vector<dynet::Expression> losses(n_words);
    unsigned prev_label = root_pos_id;
    for (unsigned i = 0; i < n_words; ++i) {
      auto payload = word_rnn.get_output(i);
      dynet::Expression logits = dense.get_output(dynet::rectify(
        merge.get_output(payload.first, payload.second, pos_embed.embed(prev_label))
      ));
      losses[i] = dynet::pickneglogsoftmax(logits, labels[i]);
      prev_label = labels[i];
    }

    return dynet::sum(losses);
  }
};

typedef CharacterRNNWithClusterPostagModel<dynet::GRUBuilder> CharacterGRUWithClusterPostagModel;
typedef CharacterRNNWithClusterPostagModel<dynet::CoupledLSTMBuilder> CharacterLSTMWithClusterPostagModel;

}

#endif  //  end for __TWPIPE_CHAR_WCLUSTER_POSTAG_MODEL_H__
