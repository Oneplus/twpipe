#ifndef LOGGING_UTILS_H
#define LOGGING_UTILS_H

#include <boost/log/trivial.hpp>
#define _TRACE BOOST_LOG_TRIVIAL(trace)
#define _INFO  BOOST_LOG_TRIVIAL(info)
#define _WARN  BOOST_LOG_TRIVIAL(warning)
#define _ERROR BOOST_LOG_TRIVIAL(error)

namespace twpipe {

void init_boost_log(bool verbose);

}

#endif  //  end for LOGGING_UTILS_H
