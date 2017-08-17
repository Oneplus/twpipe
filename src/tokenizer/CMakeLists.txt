include_directories (${PROJECT_SOURCE_DIR}/src)

add_library (twpipe_tokenizer
    tokenizer_trainer.h
    tokenizer_trainer.cc
    tokenize_model.h
    tokenize_model.cc
    tokenize_model_builder.h
    tokenize_model_builder.cc
    lin_rnn_tokenize_model.h
    seg_rnn_tokenize_model.h
    )

target_link_libraries (twpipe_tokenizer
    ${LIBS}
    dynet
    twpipe_utils)
