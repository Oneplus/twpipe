include_directories (${PROJECT_SOURCE_DIR}/src)

add_library(twpipe_parser
    tree.h
    noisify.cc
    noisify.h
    #evaluate.cc
    #evaluate.h
    state.h
    state.cc
    arcstd.cc
    arcstd.h
    #arceager.cc
    #arceager.h
    archybrid.cc
    archybrid.h
    #swap.cc
    #swap.h
    system.h
    #system_builder.cc
    #system_builder.h
    parse_model.cc
    parse_model.h
    parse_model_ballesteros15.cc
    parse_model_ballesteros15.h
    parse_model_dyer15.cc
    parse_model_dyer15.h
    parse_model_kiperwasser16.cc
    parse_model_kiperwasser16.h
    parse_model_builder.cc
    parse_model_builder.h
    parser_trainer.cc
    parser_trainer.h
    #parser_trainer.h
    #parser_trainer.cc
    )

target_link_libraries (twpipe_parser
    ${LIBS}
    dynet
    twpipe_utils)
