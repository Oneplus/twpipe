#ifndef __TWPIPE_LAYER_H__
#define __TWPIPE_LAYER_H__

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"

namespace twpipe {

struct LayerI {
  bool trainable;
  LayerI(bool trainable) : trainable(trainable) {}
  void active_training() { trainable = true; }
  void inactive_training() { trainable = false; }
  // Initialize parameter
  virtual void new_graph(dynet::ComputationGraph& cg) = 0;
};

struct SymbolEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;

  SymbolEmbedding(dynet::ParameterCollection& m,
                  unsigned n,
                  unsigned dim,
                  bool trainable = true);
  void new_graph(dynet::ComputationGraph& cg) override;
  dynet::Expression embed(unsigned label_id);
};

struct BinnedDistanceEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;
  unsigned max_bin;

  BinnedDistanceEmbedding(dynet::ParameterCollection& m,
                          unsigned hidden,
                          unsigned n_bin = 8,
                          bool trainable = true);
  void new_graph(dynet::ComputationGraph& cg) override;
  dynet::Expression embed(int distance);
};

struct BinnedDurationEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;
  unsigned max_bin;

  BinnedDurationEmbedding(dynet::ParameterCollection& m,
                          unsigned hidden,
                          unsigned n_bin = 8,
                          bool trainable = true);
  void new_graph(dynet::ComputationGraph& cg) override;
  dynet::Expression embed(unsigned dur);
};

typedef std::pair<dynet::Expression, dynet::Expression> BiRNNOutput;

template<typename RNNBuilderType>
struct RNNLayer : public LayerI {
  unsigned n_items;
  RNNBuilderType rnn;
  dynet::Parameter p_guard;
  dynet::Expression guard;
  bool reversed;

  RNNLayer(dynet::ParameterCollection& ParameterCollection,
           unsigned n_layers,
           unsigned dim_input,
           unsigned dim_hidden,
           bool rev = false,
           bool have_guard = true,
           bool trainable = true) :
    LayerI(trainable),
    n_items(0),
    rnn(n_layers, dim_input, dim_hidden, &ParameterCollection),
    p_guard(ParameterCollection.add_parameters({ dim_input, 1 })),
    reversed(rev) {
  }

  void add_inputs(const std::vector<dynet::Expression>& inputs) {
    n_items = inputs.size();
    rnn.start_new_sequence();
    if (have_guard) { rnn.add_input(guard); }
    if (reversed) {
      for (int i = n_items - 1; i >= 0; --i) { rnn.add_input(inputs[i]); }
    } else {
      for (unsigned i = 0; i < n_items; ++i) { rnn.add_input(inputs[i]); }
    }
  }

  dynet::Expression get_output(dynet::ComputationGraph* hg, int index) {
    if (reversed) { return rnn.get_h(dynet::RNNPointer(n_items - index)).back(); }
    return rnn.get_h(dynet::RNNPointer(index + 1)).back();
  }

  void get_outputs(dynet::ComputationGraph* hg,
                   std::vector<dynet::Expression>& outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) { outputs[i] = get_output(hg, i); }
  }

  dynet::Expression get_final() {
    return rnn.back();
  }

  void new_graph(dynet::ComputationGraph& hg) {
    if (!trainable) {
      std::cerr << "WARN: not-trainable RNN is not implemented." << std::endl;
    }
    rnn.new_graph(hg);
    guard = dynet::expr::parameter(hg, p_guard);
  }
  void set_dropout(float& rate) { rnn.set_dropout(rate); }
  void disable_dropout() { rnn.disable_dropout(); }
};

template<typename RNNBuilderType>
struct BiRNNLayer : public LayerI {
  unsigned n_items;
  bool have_guard;
  RNNBuilderType fw_rnn;
  RNNBuilderType bw_rnn;
  dynet::Parameter p_fw_guard;
  dynet::Parameter p_bw_guard;
  dynet::Expression fw_guard;
  dynet::Expression bw_guard;
  std::vector<dynet::Expression> fw_hidden;
  std::vector<dynet::Expression> bw_hidden;

  BiRNNLayer(dynet::ParameterCollection& model,
             unsigned n_layers,
             unsigned dim_input,
             unsigned dim_hidden,
             bool have_guard = true,
             bool trainable = true) :
    LayerI(trainable),
    n_items(0),
    fw_rnn(n_layers, dim_input, dim_hidden, model),
    bw_rnn(n_layers, dim_input, dim_hidden, model),
    p_fw_guard(model.add_parameters({ dim_input, 1 })),
    p_bw_guard(model.add_parameters({ dim_input, 1 })) {
  }

  void add_inputs(const std::vector<dynet::Expression>& inputs) {
    n_items = inputs.size();
    fw_rnn.start_new_sequence();
    bw_rnn.start_new_sequence();
    fw_hidden.resize(n_items);
    bw_hidden.resize(n_items);

    if (have_guard) { fw_rnn.add_input(fw_guard); }
    for (unsigned i = 0; i < n_items; ++i) {
      fw_hidden[i] = fw_rnn.add_input(inputs[i]);
      bw_hidden[n_items - i - 1] = bw_rnn.add_input(inputs[n_items - i - 1]);
    }
    if (have_guard) { bw_rnn.add_input(bw_guard); }
  }

  BiRNNOutput get_output(int index) {
    return std::make_pair(fw_hidden[index], bw_hidden[index]);
  }

  BiRNNOutput get_final() {
    return std::make_pair(fw_hidden[n_items - 1], bw_hidden[0]);
  }

  void get_outputs(std::vector<BiRNNOutput>& outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) {
      outputs[i] = get_output(i);
    }
  }

  void new_graph(dynet::ComputationGraph& hg) {
    if (!trainable) {
      std::cerr << "WARN: not-trainable RNN is not implemented." << std::endl;
    }
    fw_rnn.new_graph(hg);
    bw_rnn.new_graph(hg);
    fw_guard = dynet::parameter(hg, p_fw_guard);
    bw_guard = dynet::parameter(hg, p_bw_guard);
  }

  void set_dropout(float& rate) {
    fw_rnn.set_dropout(rate);
    bw_rnn.set_dropout(rate);
  }

  void disable_dropout() {
    fw_rnn.disable_dropout();
    bw_rnn.disable_dropout();
  }
};

struct InputLayer : public LayerI {
  dynet::ComputationGraph * _cg;
  unsigned dim;
  InputLayer(unsigned dim) : LayerI(false), dim(dim) {}

  void new_graph(dynet::ComputationGraph& cg) {
    _cg = &cg;
  }

  dynet::Expression get_output(const std::vector<float> & data) {
    return dynet::input(*_cg, { dim }, data);
  }
};

struct SoftmaxLayer : public LayerI {
  dynet::Parameter p_B, p_W;
  dynet::Expression B, W;

  SoftmaxLayer(dynet::ParameterCollection& model,
               unsigned dim_input,
               unsigned dim_output,
               bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;

  dynet::Expression get_output(const dynet::Expression & expr);
};

struct DenseLayer : public LayerI {
  dynet::Parameter p_W, p_B;
  dynet::Expression W, B;

  DenseLayer(dynet::ParameterCollection& model,
             unsigned dim_input,
             unsigned dim_output,
             bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;

  dynet::Expression get_output(const dynet::Expression & expr);
};

struct Merge2Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2;
  dynet::Expression B, W1, W2;

  Merge2Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;

  dynet::Expression get_output(const dynet::Expression & expr1,
                               const dynet::Expression & expr2);
};

struct Merge3Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3;
  dynet::Expression B, W1, W2, W3;

  Merge3Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;

  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3);
};

struct Merge4Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4;
  dynet::Expression B, W1, W2, W3, W4;

  Merge4Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;

  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3,
                               const dynet::Expression& expr4);
};

struct Merge5Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4, p_W5;
  dynet::Expression B, W1, W2, W3, W4, W5;

  Merge5Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_input5,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;

  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3,
                               const dynet::Expression& expr4,
                               const dynet::Expression& expr5);
};

struct Merge6Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4, p_W5, p_W6;
  dynet::Expression B, W1, W2, W3, W4, W5, W6;

  Merge6Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_input5,
              unsigned dim_input6,
              unsigned dim_output,
              bool trainable = true);
  void new_graph(dynet::ComputationGraph& hg) override;
  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3,
                               const dynet::Expression& expr4,
                               const dynet::Expression& expr5,
                               const dynet::Expression& expr6);
};

template <class RNNBuilderType>
struct SegUniEmbedding {
  // uni-directional segment embedding.
  dynet::Parameter p_h0;
  RNNBuilderType builder;
  std::vector<std::vector<dynet::Expression>> h;
  unsigned len;

  explicit SegUniEmbedding(dynet::ParameterCollection& m,
                           unsigned n_layers,
                           unsigned rnn_input_dim,
                           unsigned seg_dim) :
    p_h0(m.add_parameters({ rnn_input_dim })),
    builder(n_layers, rnn_input_dim, seg_dim) {
  }

  void construct_chart(dynet::ComputationGraph& cg,
                       const std::vector<dynet::Expression>& c,
                       int max_seg_len = 0) {
    len = c.size();
    h.clear(); // The first dimension for h is the starting point, the second is length.
    h.resize(len);
    dynet::expr::Expression h0 = dynet::expr::parameter(cg, p_h0);
    builder.new_graph(cg);
    for (unsigned i = 0; i < len; ++i) {
      unsigned max_j = i + len;
      if (max_seg_len) { max_j = i + max_seg_len; }
      if (max_j > len) { max_j = len; }
      unsigned seg_len = max_j - i;
      auto& hi = h[i];
      hi.resize(seg_len);

      builder.start_new_sequence();
      builder.add_input(h0);
      // Put one span in h[i][j]
      for (unsigned k = 0; k < seg_len; ++k) {
        hi[k] = builder.add_input(c[i + k]);
      }
    }
  }

  const dynet::Expression& operator()(unsigned i, unsigned j) const {
    BOOST_ASSERT(j <= len);
    BOOST_ASSERT(j >= i);
    return h[i][j - i];
  }

  void set_dropout(float& rate) {
    builder.set_dropout(rate);
  }

  void disable_dropout() {
    builder.disable_dropout();
  }
};

template <class RNNBuilderType>
struct SegBiEmbedding {
  typedef std::pair<dynet::Expression, dynet::Expression> ExpressionPair;
  SegUniEmbedding<RNNBuilderType> fwd, bwd;
  std::vector<std::vector<ExpressionPair>> h;
  unsigned len;

  explicit SegBiEmbedding(dynet::ParameterCollection& m,
                          unsigned n_layers,
                          unsigned rnn_input_dim,
                          unsigned seg_dim) :
    fwd(m, n_layers, rnn_input_dim, seg_dim),
    bwd(m, n_layers, rnn_input_dim, seg_dim) {
  }

  void construct_chart(dynet::ComputationGraph& cg,
                       const std::vector<dynet::Expression>& c,
                       int max_seg_len = 0) {
    len = c.size();
    fwd.construct_chart(cg, c, max_seg_len);

    std::vector<dynet::Expression> rc(len);
    for (unsigned i = 0; i < len; ++i) { rc[i] = c[len - i - 1]; }
    bwd.construct_chart(cg, rc, max_seg_len);

    h.clear();
    h.resize(len);
    for (unsigned i = 0; i < len; ++i) {
      unsigned max_j = i + len;
      if (max_seg_len) { max_j = i + max_seg_len; }
      if (max_j > len) { max_j = len; }
      auto& hi = h[i];
      unsigned seg_len = max_j - i;
      hi.resize(seg_len);
      for (unsigned k = 0; k < seg_len; ++k) {
        unsigned j = i + k;
        const dynet::expr::Expression& fe = fwd(i, j);
        const dynet::expr::Expression& be = bwd(len - 1 - j, len - 1 - i);
        hi[k] = std::make_pair(fe, be);
      }
    }
  }

  const ExpressionPair& operator()(unsigned i, unsigned j) const {
    BOOST_ASSERT(j <= len);
    BOOST_ASSERT(j >= i);
    return h[i][j - i];
  }

  void set_dropout(float& rate) {
    fwd.set_dropout(rate);
    bwd.set_dropout(rate);
  }

  void disable_dropout() {
    fwd.disable_dropout();
    bwd.disable_dropout();
  }
};

}

#endif  //  end for LAYER_H
