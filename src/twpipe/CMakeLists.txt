add_library (twpipe_utils
    logging.h
    logging.cc
    alphabet.h
    alphabet.cc
    alphabet_collection.h
    alphabet_collection.cc
    corpus.h
    corpus.cc
    optimizer_builder.h
    optimizer_builder.cc
    trainer.h
    trainer.cc
    model.h
    model.cc
    elmo.h
    elmo.cc
    embedding.h
    embedding.cc
    cluster.h
    cluster.cc
    normalizer.h
    normalizer.cc
    ensemble.h
    ensemble.cc
    math.h
    math.cc
    unicode.h
    unicode.cc
    )

target_link_libraries(twpipe_utils ${LIBS})
