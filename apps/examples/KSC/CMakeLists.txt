#-------------------------------------------------------------------------------
# Set cmake minimum version
#
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)
project(app-ksc)

#-------------------------------------------------------------------------------
# Find the `leuven` library
#
find_package (leuven CONFIG REQUIRED)
if (NOT leuven_FOUND)
  message (FATAL_ERROR "`leuven` library was not found")
endif ()

find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
# Set the same CXX flags that were used when building the `leuven` library
#
set (CMAKE_CXX_FLAGS "${leuven_CMAKE_CXX_FLAGS}")
message(STATUS "CXX-FLAGS: ${CMAKE_CXX_FLAGS}")


#-------------------------------------------------------------------------------
# We need Fortran compiler (linking) in case of FBLAS-wrapper
#
if (FORTRAN_COMPILER_REQUIRED)
  enable_language(Fortran)
endif (FORTRAN_COMPILER_REQUIRED)


#-------------------------------------------------------------------------------
# Locate sources and headers for this project (add `leuven` include DIR as well)
#
include_directories (${PROJECT_SOURCE_DIR}/include
                     ${leuven_INCLUDE_DIR})
file (GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file (GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)


#-------------------------------------------------------------------------------
# Add the executable, and link it to the `leuven` libraries (all dependencies)
#
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#
add_executable (KscIchol_Train   KscIchol_Train.cc   ${sources} ${headers})
add_executable (KscIchol_Tune    KscIchol_Tune.cc    ${sources} ${headers})
add_executable (KscIchol_Test    KscIchol_Test.cc    ${sources} ${headers})

add_executable (Ksc_Train   Ksc_Train.cc   ${sources} ${headers})
add_executable (Ksc_Tune    Ksc_Tune.cc   ${sources} ${headers})
add_executable (Ksc_Test    Ksc_Test.cc   ${sources} ${headers})

#
target_link_libraries (KscIchol_Train    ${leuven_LIBRARIES})
target_link_libraries (KscIchol_Tune     ${leuven_LIBRARIES})
target_link_libraries (KscIchol_Test     ${leuven_LIBRARIES})

target_link_libraries (Ksc_Train    ${leuven_LIBRARIES})
target_link_libraries (Ksc_Tune     ${leuven_LIBRARIES})
target_link_libraries (Ksc_Test     ${leuven_LIBRARIES})


#-------------------------------------------------------------------------------
# In case of fortran compiler i.e. when FBLAS wrapper was used to build the lib.
#
if (FORTRAN_COMPILER_REQUIRED)
  set_property(TARGET KscIchol_Train   PROPERTY LINKER_LANGUAGE Fortran)
  set_property(TARGET KscIchol_Tune    PROPERTY LINKER_LANGUAGE Fortran)
  set_property(TARGET KscIchol_Test    PROPERTY LINKER_LANGUAGE Fortran)

  set_property(TARGET Ksc_Train   PROPERTY LINKER_LANGUAGE Fortran)
  set_property(TARGET Ksc_Tune    PROPERTY LINKER_LANGUAGE Fortran)
  set_property(TARGET Ksc_Test    PROPERTY LINKER_LANGUAGE Fortran)

endif (FORTRAN_COMPILER_REQUIRED)


#-------------------------------------------------------------------------------
# Install the executables (with RPATH)
#
install (TARGETS KscIchol_Train KscIchol_Tune KscIchol_Test Ksc_Train Ksc_Tune Ksc_Test RUNTIME DESTINATION bin)
install (DIRECTORY tests DESTINATION .)
