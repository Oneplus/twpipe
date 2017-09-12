include_directories (${PROJECT_SOURCE_DIR}/src)

add_library(twpipe_parser
    tree.h
    noisify.cc
    noisify.h
    state.h
    state.cc
    arcstd.cc
    arcstd.h
    arceager.cc
    arceager.h
    archybrid.cc
    archybrid.h
    swap.cc
    swap.h
    system.h
    system.cc
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
    )

target_link_libraries (twpipe_parser
    ${LIBS}
    dynet
    dynet_layer
    twpipe_utils)

ADD_EXECUTABLE (generate_ensemble_data
	generate_ensemble_data.cc
	ensemble_generator.h
	ensemble_generator.cc)

TARGET_LINK_LIBRARIES (generate_ensemble_data ${LIBS} twpipe_parser twpipe_utils)
