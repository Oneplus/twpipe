#include "sys_utils.h"
#include "logging.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#if _MSC_VER
#include <process.h>
#endif


int portable_getpid() {
#ifdef _MSC_VER
  return _getpid();
#else
  return getpid();
#endif
}


float execute_and_get_result(const std::string& cmd_prefix, const std::string& output) {
  std::string cmd = cmd_prefix + " " + output;
  _TRACE << "Running: " << cmd;
#ifndef _MSC_VER
  FILE* pipe = popen(cmd.c_str(), "r");
#else
  FILE* pipe = _popen(cmd.c_str(), "r");
#endif
  if (!pipe) {
    return 0.;
  }
  char buffer[128];
  std::string result = "";
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != NULL) { result += buffer; }
  }
#ifndef _MSC_VER
  pclose(pipe);
#else
  _pclose(pipe);
#endif

  std::stringstream S(result);
  std::string token;
  while (S >> token) {
    boost::algorithm::trim(token);
    return boost::lexical_cast<double>(token);
  }
  return 0.;
}
