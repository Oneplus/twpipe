include_directories (${PROJECT_SOURCE_DIR}/src)

add_library (twpipe_postagger
    postagger_trainer.h
    postagger_trainer.cc
    postag_model.h
    postag_model.cc
    postag_model_builder.h
    postag_model_builder.cc
    char_rnn_postag_model.h
    char_rnn_crf_postag_model.h
    char_rnn_wcluster_postag_model.h
    word_rnn_postag_model.h
    word_char_rnn_postag_model.h
    )

target_link_libraries (twpipe_postagger
    ${LIBS}
    dynet
    dynet_layer
    twpipe_utils)
