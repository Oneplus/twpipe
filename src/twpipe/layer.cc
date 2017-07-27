#include "layer.h"
#include "dynet/param-init.h"
#include <boost/assert.hpp>

namespace twpipe {

SymbolEmbedding::SymbolEmbedding(dynet::ParameterCollection& m,
                                 unsigned n,
                                 unsigned dim,
                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n, { dim, 1 })) {
}

void SymbolEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::Expression SymbolEmbedding::embed(unsigned index) {
  return (trainable ?
          dynet::lookup((*cg), p_e, index) :
          dynet::const_lookup((*cg), p_e, index));
}

BinnedDistanceEmbedding::BinnedDistanceEmbedding(dynet::ParameterCollection& m,
                                                 unsigned dim,
                                                 unsigned n_bins,
                                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n_bins * 2, { dim, 1 })),
  max_bin(n_bins - 1) {
  BOOST_ASSERT_MSG(n_bins > 0, "Layer: number of bins should be larger than zero.");
}

void BinnedDistanceEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::Expression BinnedDistanceEmbedding::embed(int dist) {
  unsigned base = (dist < 0 ? max_bin : 0);
  unsigned dist_std = 0;
  if (dist) {
    dist_std = static_cast<unsigned>(log(dist < 0 ? -dist : dist) / log(1.6f)) + 1;
  }
  if (dist_std > max_bin) {
    dist_std = max_bin;
  }
  return (trainable ?
          dynet::lookup(*cg, p_e, dist_std + base) :
          dynet::const_lookup(*cg, p_e, dist_std + base));
}

BinnedDurationEmbedding::BinnedDurationEmbedding(dynet::ParameterCollection& m,
                                                 unsigned dim,
                                                 unsigned n_bins,
                                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n_bins, { dim, 1 })),
  max_bin(n_bins - 1) {
  BOOST_ASSERT_MSG(n_bins > 0, "Layer: number of bins should be larger than zero.");
}

void BinnedDurationEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::Expression BinnedDurationEmbedding::embed(unsigned dur) {
  if (dur) {
    dur = static_cast<unsigned>(log(dur) / log(1.6f)) + 1;
  }
  if (dur > max_bin) {
    dur = max_bin;
  }
  return (trainable ?
          dynet::lookup((*cg), p_e, dur) :
          dynet::const_lookup((*cg), p_e, dur));
}

SoftmaxLayer::SoftmaxLayer(dynet::ParameterCollection& m,
                           unsigned dim_input,
                           unsigned dim_output,
                           bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W(m.add_parameters({ dim_output, dim_input })) {
}

void SoftmaxLayer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W = dynet::parameter(hg, p_W);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W = dynet::const_parameter(hg, p_W);
  }
}

dynet::Expression SoftmaxLayer::get_output(const dynet::Expression& expr) {
  return dynet::log_softmax(dynet::affine_transform({ B, W, expr }));
}

DenseLayer::DenseLayer(dynet::ParameterCollection& m,
                       unsigned dim_input,
                       unsigned dim_output,
                       bool trainable) :
  LayerI(trainable),
  p_W(m.add_parameters({ dim_output, dim_input })),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))) {
}

void DenseLayer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    W = dynet::parameter(hg, p_W);
    B = dynet::parameter(hg, p_B);
  } else {
    W = dynet::const_parameter(hg, p_W);
    B = dynet::const_parameter(hg, p_B);
  }
}

dynet::Expression DenseLayer::get_output(const dynet::Expression& expr) {
  return dynet::affine_transform({ B, W, expr });
}

Merge2Layer::Merge2Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })) {
}

void Merge2Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
  }
}

dynet::Expression Merge2Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2) {
  return dynet::affine_transform({ B, W1, expr1, W2, expr2 });
}

Merge3Layer::Merge3Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })) {
}

void Merge3Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
  }
}

dynet::Expression Merge3Layer::get_output(const dynet::Expression& expr1,
                                                  const dynet::Expression& expr2,
                                                  const dynet::Expression& expr3) {
  return dynet::affine_transform({ B, W1, expr1, W2, expr2, W3, expr3 });
}

Merge4Layer::Merge4Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })) {
}

void Merge4Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
    W4 = dynet::parameter(hg, p_W4);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
    W4 = dynet::const_parameter(hg, p_W4);
  }
}

dynet::Expression Merge4Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2,
                                          const dynet::Expression& expr3,
                                          const dynet::Expression& expr4) {
  return dynet::affine_transform({ B, W1, expr1, W2, expr2, W3, expr3, W4, expr4 });
}

Merge5Layer::Merge5Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })),
  p_W5(m.add_parameters({ dim_output, dim_input5 })) {
}

void Merge5Layer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
    W4 = dynet::parameter(hg, p_W4);
    W5 = dynet::parameter(hg, p_W5);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
    W4 = dynet::const_parameter(hg, p_W4);
    W5 = dynet::const_parameter(hg, p_W5);
  }
}

dynet::Expression Merge5Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2,
                                          const dynet::Expression& expr3,
                                          const dynet::Expression& expr4,
                                          const dynet::Expression& expr5) {
  return dynet::affine_transform({
    B, W1, expr1, W2, expr2, W3, expr3, W4, expr4, W5, expr5
  });
}

Merge6Layer::Merge6Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_input6,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })),
  p_W5(m.add_parameters({ dim_output, dim_input5 })),
  p_W6(m.add_parameters({ dim_output, dim_input6 })) {
}

void Merge6Layer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
    W4 = dynet::parameter(hg, p_W4);
    W5 = dynet::parameter(hg, p_W5);
    W6 = dynet::parameter(hg, p_W6);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
    W4 = dynet::const_parameter(hg, p_W4);
    W5 = dynet::const_parameter(hg, p_W5);
    W6 = dynet::const_parameter(hg, p_W6);
  }
}

dynet::Expression Merge6Layer::get_output(
  const dynet::Expression& expr1,
  const dynet::Expression& expr2,
  const dynet::Expression& expr3,
  const dynet::Expression& expr4,
  const dynet::Expression& expr5,
  const dynet::Expression& expr6) {
  return dynet::affine_transform({
    B, W1, expr1, W2, expr2, W3, expr3, W4, expr4, W5, expr5, W6, expr6
  });
}

}