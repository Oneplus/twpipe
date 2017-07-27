add_subdirectory (twpipe)
add_subdirectory (tokenizer)
add_subdirectory (postagger)

add_executable (twpipe twpipe.cc)
target_link_libraries (twpipe
    ${LIBS}
    dynet
    twpipe_utils
    twpipe_tokenizer
    twpipe_postagger)