#include "evaluate.h"
#include "logging.h"
#include "sys_utils.h"
#include "tree.h"
#include <fstream>
#include <chrono>

float evaluate(const po::variables_map& conf,
               Corpus& corpus,
               Parser& parser,
               const std::string& output,
               bool labelling_only) {
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);

  std::ofstream ofs(output);
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    InputUnits& input_units = corpus.devel_inputs[sid];
    const ParseUnits& parse = corpus.devel_parses[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.training_vocab.count(u.wid)) { u.wid = kUNK; }
    }
    dynet::ComputationGraph cg;
    ParseUnits output;
    if (labelling_only) {
      if (!parser.sys.allow_nonprojective() && DependencyUtils::is_non_projective(parse)) {
        output = parse;
      } else {
        parser.label(cg, input_units, parse, output);
      }
    } else {
      parser.predict(cg, input_units, output);
    }

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }

    unsigned len = input_units.size();
    // pay attention to this, not counting the last DUMMY_ROOT
    for (unsigned i = 0; i < len - 1; ++i) {
      ofs << i + 1 << "\t"                //  id
        << input_units[i].w_str << "\t"   //  form
        << input_units[i].l_str << "\t"   //  lemma
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << input_units[i].f_str << "\t"
        ;

      if (parse[i].head == Corpus::REMOVED_HED) {
        ofs << "-3\t_\t";
      } else {
        ofs << (parse[i].head >= len ? 0 : parse[i].head) << "\t"
          << corpus.deprel_map.get(parse[i].deprel) << "\t";
      }

      if (output[i].head == Corpus::REMOVED_HED) {
        ofs << "-3\t_";
      } else {
        ofs << (output[i].head >= len ? 0 : output[i].head) << "\t"
          << corpus.deprel_map.get(output[i].deprel);
      }
      ofs << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  float f_score = execute_and_get_result(conf["external_eval"].as<std::string>(),
                                         output);
  _INFO << "Evaluate:: UAS " << f_score << " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

float beam_search(const po::variables_map & conf,
                  Corpus & corpus,
                  Parser & parser,
                  const std::string & output) {
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  unsigned beam_size = conf["beam_size"].as<unsigned>();
  bool structure = (conf["supervised_objective"].as<std::string>() == "structure");
  
  std::ofstream ofs(output);
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    InputUnits& input_units = corpus.devel_inputs[sid];
    const ParseUnits& parse = corpus.devel_parses[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.training_vocab.count(u.wid)) { u.wid = kUNK; }
    }
    dynet::ComputationGraph cg;
    std::vector<ParseUnits> results;
    parser.beam_search(cg, input_units, beam_size, structure, results);

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }

    ParseUnits & output = results[0];
    unsigned len = input_units.size();
    for (unsigned i = 0; i < len - 1; ++i) {
      ofs << i + 1 << "\t"                //  id
        << input_units[i].w_str << "\t"   //  form
        << input_units[i].n_str << "\t"   //  lemma
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << corpus.pos_map.get(input_units[i].pid) << "\t"
        << input_units[i].f_str << "\t"
        ;

      if (parse[i].head == Corpus::REMOVED_HED) {
        ofs << "-3\t_\t";
      } else {
        ofs << (parse[i].head >= len ? 0 : parse[i].head) << "\t"
          << corpus.deprel_map.get(parse[i].deprel) << "\t";
      }

      if (output[i].head == Corpus::REMOVED_HED) {
        ofs << "-3\t_";
      } else {
        ofs << (output[i].head >= len ? 0 : output[i].head) << "\t"
          << corpus.deprel_map.get(output[i].deprel);
      }
      ofs << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  float f_score = execute_and_get_result(conf["external_eval"].as<std::string>(),
                                         output);
  _INFO << "Evaluate:: UAS " << f_score << " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}
