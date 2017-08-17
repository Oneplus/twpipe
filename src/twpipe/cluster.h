#ifndef __TWPIPE_CLUSTER_H__
#define __TWPIPE_CLUSTER_H__

#include <iostream>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

struct WordCluster {
protected:
  static WordCluster * instance;
  std::unordered_map<std::string, std::string> cluster;

  WordCluster();

public:
  static po::options_description get_options();

  static WordCluster* get();

  void load(const std::string & cluster_file);

  void empty();

  void render(const std::vector<std::string> & words, 
              std::vector<std::string> & values);
};

}

#endif  //  end for __TWPIPE_CLUSTER_H__