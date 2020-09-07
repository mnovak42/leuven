################################################################################
# Add sub-directories
################################################################################

message (STATUS "${Green}===== Adding the lssvm part ===== ${ColourReset}")


################################################################################
# Add some project sources headers and set include directories
#
file (GLOB CXXHEADERS 
  ${CMAKE_SOURCE_DIR}/lssvm/utils/inc/*.hh 
  ${CMAKE_SOURCE_DIR}/lssvm/utils/inc/*.tpp
  ${CMAKE_SOURCE_DIR}/lssvm/ksc/inc/*.hh 
  ${CMAKE_SOURCE_DIR}/lssvm/ksc/inc/*.tpp 
)

file (GLOB CXXSOURCES 
  ${CMAKE_SOURCE_DIR}/lssvm/utils/src/*.cc
  ${CMAKE_SOURCE_DIR}/lssvm/ksc/src/*.cc
)

include_directories (
  ${CMAKE_SOURCE_DIR}/utils/inc
  ${CMAKE_SOURCE_DIR}/utils/inc/global
  ${CMAKE_SOURCE_DIR}/lssvm/utils/inc
  ${CMAKE_SOURCE_DIR}/lssvm/ksc/inc
)

message ("")



################################################################################
# Add target library and links
#
# NOTE: PURE C++ i.e. no CUDA. I will change thispart according to leuven-lib
#       when I will have CUDA sources here
#
#add_library(ksc STATIC ${CXXSOURCES})
#target_link_libraries(ksc ${PROJECTNAME} ${CPU_BLAS_LIBRARY}) 



################################################################################
# Install: headers, library 
#
install (FILES ${CXXHEADERS} DESTINATION includes)
#install (TARGETS ksc DESTINATION lib)

