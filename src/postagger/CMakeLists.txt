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

add_executable (generate_postag_ensemble_data
    generate_ensemble_data.cc
    ensemble_generator.h
    ensemble_generator.cc
    )

target_link_libraries (generate_postag_ensemble_data
    ${LIBS}
    dynet
    dynet_layer
    twpipe_utils)
