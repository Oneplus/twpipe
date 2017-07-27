include_directories (${PROJECT_SOURCE_DIR}/src)

add_library (twpipe_postagger
    postagger_trainer.h
    postagger_trainer.cc
    postag_model.h
    postag_model.cc
    postag_model_builder.h
    postag_model_builder.cc
    char_rnn_postag_model.h
    )
