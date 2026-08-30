# TWT+PowerCast Controller Python Bindings Module CMakeLists.txt
# Author: Ahmed Maksud <amaks002@ucr.edu>
# Lab: SHINE Lab, Texas State University
# PI: Marcelo Menezes De Carvalho
# This module provides Python-C++ communication interface for the
# TWT+PowerCast Controller (802.11ax TWT scheduling + P21XXCSR-EVB RF energy
# harvesting). Self-contained for future repo extraction.

cmake_minimum_required(VERSION 3.10)

# Project information
set(PROJECT_AUTHOR "Ahmed Maksud")
set(PROJECT_EMAIL "amaks002@ucr.edu")
set(PROJECT_LAB "SHINE Lab, Texas State University")
set(PROJECT_PI "Marcelo Menezes De Carvalho")
set(PROJECT_DESCRIPTION "NS3-AI TWT+PowerCast Controller Python Binding Interface")

# Python Bindings Library Sources with pb-twt- prefix
set(TWT_PYTHON_BINDINGS_SOURCES
    pb-twt-wrapper-ns3.cc
)

set(TWT_PYTHON_BINDINGS_HEADERS
    pb-twt-wrapper-ns3.h
    pb-twt-core.h
)

# PowerCast Energy Harvesting sources (copied from powercast_hardware/ to keep
# rl-twt-powercast/ self-contained for future repo extraction).
set(POWERCAST_HARVESTER_SOURCES
    ph-harvester-hardware.cc
    ph-deployment-helper.cc
)

set(POWERCAST_HARVESTER_HEADERS
    ph-harvester-hardware.h
    ph-deployment-helper.h
)

# Print author information
message(STATUS "========================================")
message(STATUS "  TWT+PowerCast Controller Python Bindings Module")
message(STATUS "  Author: ${PROJECT_AUTHOR}")
message(STATUS "  Email: ${PROJECT_EMAIL}")
message(STATUS "  Lab: ${PROJECT_LAB}")
message(STATUS "  PI: ${PROJECT_PI}")
message(STATUS "  Description: ${PROJECT_DESCRIPTION}")
message(STATUS "========================================")

# Build TWT+PowerCast main simulation with modular components + harvester sources
build_lib_example(
  NAME twt-powercast-main-simulation
  SOURCE_FILES
    twt-powercast-main-simulation.cc
    twt-simulation-config.cc
    twt-trace-callbacks.cc
    twt-metrics.cc
    pb-twt-wrapper-ns3.cc
    ph-harvester-hardware.cc
    ph-deployment-helper.cc
  LIBRARIES_TO_LINK
    ${libai}
    ${libcore}
    ${libnetwork}
    ${libwifi}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libspectrum}
    ${libpropagation}
    ${libantenna}
    ${libenergy}
    ${libflow-monitor}
    ${libpoint-to-point}
    ${libinternet-apps}
)

# Python binding module using pybind11 (renamed to avoid colliding with twt/)
pybind11_add_module(pb_twt_powercast_interface_py pb-twt-interface.cc)
set_target_properties(pb_twt_powercast_interface_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_NAME "pb_twt_powercast_interface_py"
    PREFIX ""
    SUFFIX ".so"
)
target_link_libraries(pb_twt_powercast_interface_py PRIVATE ${libai})

# Deploy .so to both the cmake-cache build dir and the source dir.
# CANONICAL directory name -- this repo must be cloned as "rl-twt-powercast"
# (see README "Quick start"); paths are hardcoded to it project-wide. Verified
# here rather than assumed: a mismatch would copy the binding to a build path
# nobody imports from, and every Python entry point would fail with
# ModuleNotFoundError with no hint at the cause.
set(_ex_name "rl-twt-powercast")
get_filename_component(_actual_ex_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
if(NOT _actual_ex_name STREQUAL _ex_name)
    message(FATAL_ERROR
        "This example must live in a directory named '${_ex_name}', but it is in "
        "'${_actual_ex_name}'. Rename it (git clone <repo-url> ${_ex_name}) and "
        "update add_subdirectory() in contrib/ai/examples/CMakeLists.txt.")
endif()
add_custom_command(TARGET pb_twt_powercast_interface_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:pb_twt_powercast_interface_py>
    ${CMAKE_BINARY_DIR}/contrib/ai/examples/${_ex_name}/pb_twt_powercast_interface_py.so

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:pb_twt_powercast_interface_py>
    ${CMAKE_CURRENT_SOURCE_DIR}/pb_twt_powercast_interface_py.so

    COMMENT "Copying TWT+PowerCast Python binding library by ${PROJECT_AUTHOR}"
)

# Deploy harvester config file to the build directory so the sim can find it via
# a build-relative path (matches the powercast_hardware/ pattern).
add_custom_target(deploy_twt_powercast_config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/ph-harvester-config.txt
    ${CMAKE_BINARY_DIR}/ph-harvester-config.txt

    COMMENT "Deploying TWT+PowerCast harvester configuration"
)

# Export variables for use by parent CMakeLists
set(TWT_PYTHON_BINDINGS_SOURCES ${TWT_PYTHON_BINDINGS_SOURCES} PARENT_SCOPE)
set(TWT_PYTHON_BINDINGS_HEADERS ${TWT_PYTHON_BINDINGS_HEADERS} PARENT_SCOPE)
set(TWT_PYTHON_BINDINGS_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(POWERCAST_HARVESTER_SOURCES ${POWERCAST_HARVESTER_SOURCES} PARENT_SCOPE)
set(POWERCAST_HARVESTER_HEADERS ${POWERCAST_HARVESTER_HEADERS} PARENT_SCOPE)
set(TWT_PYTHON_BINDINGS_AUTHOR "${PROJECT_AUTHOR}" PARENT_SCOPE)

# Installation rules
install(FILES ${TWT_PYTHON_BINDINGS_HEADERS} ${POWERCAST_HARVESTER_HEADERS}
    DESTINATION include/ns3/ai/twt-powercast-bindings
    COMPONENT development
)

install(TARGETS pb_twt_powercast_interface_py
    LIBRARY DESTINATION lib/twt-powercast-bindings
    COMPONENT runtime
)
