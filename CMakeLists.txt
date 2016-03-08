cmake_minimum_required(VERSION 3.1)

########
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
########

########
# General project settings
project(distgend)
# Enable support for external projects
include(ExternalProject)
########

########
# Compiler/linker options based on the different compilers
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Using Clang
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    ADD_DEFINITIONS(-g -Ofast -march=native)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    ADD_DEFINITIONS(-g3 -O0 -Weverything)
  endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # Using GCC
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    ADD_DEFINITIONS(-g -O2 -march=native)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    ADD_DEFINITIONS(-g3 -O0 -Wall -Wextra -Wunused)
  endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # Using Intel C++
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    ADD_DEFINITIONS(-g -fast)
  endif()

  if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    ADD_DEFINITIONS(-g3 -O0 -Wall -Wextra)
  endif()

endif()
########

########
# OpenMP
find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
########

########
# libponci
ExternalProject_Add(libponci
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libponci
    CONFIGURE_COMMAND  CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libponci
    BUILD_IN_SOURCE 0
    BUILD_COMMAND ${MAKE})

# Store libponci variable in directory
ExternalProject_Get_Property(libponci install_dir)
set (libponci_path ${install_dir})
include_directories(${libponci_path}/include)
link_directories(${libponci_path}/lib)
########

########
# Compiling and linking
add_executable(distgen distgen.c distgen_internal.c)
set_property(TARGET distgen PROPERTY C_STANDARD 11)

add_library(distgend distgen_internal.c distgend.c)
add_dependencies(distgend libponci)
set_property(TARGET distgend PROPERTY C_STANDARD 11)
INSTALL(TARGETS distgend DESTINATION "lib")
INSTALL(FILES distgend.h DESTINATION "include")

add_executable(distgend_example example.cpp )
add_dependencies(distgend_example libponci)
target_link_libraries(distgend_example distgend ponci)
set_property(TARGET distgend_example PROPERTY CXX_STANDARD 11)
set_property(TARGET distgend_example PROPERTY C_STANDARD 11)
########
