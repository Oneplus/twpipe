#ifndef ARCHYBRID_H
#define ARCHYBRID_H

#include "system.h"

struct ArcHybrid : public TransitionSystem {
  unsigned n_actions;
  std::vector<std::string> action_names;

  ArcHybrid(const Alphabet& deprel_map);

  std::string name(unsigned id) const override;

  bool allow_nonprojective() const override;

  unsigned num_actions() const override;

  unsigned num_deprels() const override;

  void get_transition_costs(const State & state,
                            const std::vector<unsigned> & actions,
                            const std::vector<unsigned> & ref_heads,
                            const std::vector<unsigned> & ref_deprels,
                            std::vector<float> & rewards) override;

  void perform_action(State & state, const unsigned& action) override;

  bool is_valid_action(const State& state, const unsigned& act) const override; 
  
  void get_valid_actions(const State& state,
                         std::vector<unsigned>& valid_actions) override;

  void get_oracle_actions(const std::vector<unsigned>& heads,
                          const std::vector<unsigned>& deprels,
                          std::vector<unsigned>& actions) override;
  
  unsigned get_structure_action(const unsigned & action) override;
  
  void shift_unsafe(State& state) const;
  void drop_unsafe(State& state) const;
  void left_unsafe(State& state, const unsigned& deprel) const;
  void right_unsafe(State& state, const unsigned& deprel) const;

  float shift_dynamic_loss_unsafe(State& state,
                                  const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels) const;

  float drop_dynamic_loss_unsafe(State& state,
                                 const std::vector<unsigned>& heads,
                                 const std::vector<unsigned>& deprels) const;

  float left_dynamic_loss_unsafe(State& state,
                                 const unsigned& deprel,
                                 const std::vector<unsigned>& heads,
                                 const std::vector<unsigned>& deprels) const;

  float right_dynamic_loss_unsafe(State& state,
                                  const unsigned& deprel,
                                  const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels) const;

  static bool is_shift(const unsigned& action);
  static bool is_drop(const unsigned& action);
  static bool is_left(const unsigned& action);
  static bool is_right(const unsigned& action);

  static unsigned get_shift_id();
  static unsigned get_drop_id();
  static unsigned get_left_id(const unsigned& deprel);
  static unsigned get_right_id(const unsigned& deprel);
  static unsigned parse_label(const unsigned& action);

  void get_oracle_actions_onestep(const std::vector<unsigned>& heads,
                                  const std::vector<unsigned>& deprels,
                                  std::vector<unsigned>& sigma,
                                  unsigned& beta,
                                  std::vector<unsigned>& output,
                                  std::vector<unsigned>& actions);
};

#endif  //  end for ARCHYBRID_H