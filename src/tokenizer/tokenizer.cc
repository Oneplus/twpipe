#include "tokenizer.h"

twpipe::Tokenizer::Tokenizer(TokenizeModel * engine) : engine(engine) {
}

void twpipe::Tokenizer::tokenize(const std::string & context,
                                 std::vector<std::string>& result) {
}

void twpipe::Tokenizer::evaluate(const std::vector<std::string>& system, const std::vector<std::string>& answer) {
}
