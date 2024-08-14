######################################################################
# CMakeLists.txt generated for component flyBall
# Product type: executable
######################################################################
cmake_minimum_required(VERSION 3.12)
project(flyBall)
if(DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS} AND NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS $ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
endif()


######################################################################
## Path variables
######################################################################
get_filename_component(START_DIR .. ABSOLUTE)
set(MATLAB_ROOT D:/matlab2023a CACHE PATH "")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(BUILDING_IN_SOURCE_DIR TRUE)
    set(BINARY_START_DIR "${START_DIR}")
else()
    set(BUILDING_IN_SOURCE_DIR FALSE)
    set(BINARY_START_DIR "${CMAKE_BINARY_DIR}")
endif()
if(WIN32 AND NOT MINGW)
    set(SYSLIB_PREFIX "lib")
else()
    set(SYSLIB_PREFIX "")
endif()

######################################################################
## System libraries
######################################################################
find_library(FOUND_LIBM m NO_SYSTEM_ENVIRONMENT_PATH PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
    ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

######################################################################
## Target definition and commands
######################################################################
set(FLYBALL_COMMON_DEFINES -DMODEL=flyBall
    -DNUMST=2
    -DNCSTATES=2
    -DHAVESTDIO
    -DMODEL_HAS_DYNAMICALLY_LOADED_SFCNS=0
    -DCLASSIC_INTERFACE=0
    -DALLOCATIONFCN=0
    -DTID01EQ=1
    -DTERMFCN=1
    -DONESTEPFCN=1
    -DMAT_FILE=0
    -DMULTI_INSTANCE_CODE=1
    -DINTEGER_CODE=0
    -DMT=0)

set(FLYBALL_COMMON_LINKER_FLAGS -static
    -m64)

add_library(flyBall_objects OBJECT ${START_DIR}/flyBall_ert_rtw/flyBall.cpp
    ${START_DIR}/flyBall_ert_rtw/flyBall_data.cpp)
set_target_properties(flyBall_objects PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(flyBall_objects PUBLIC cxx_std_11)
target_compile_definitions(flyBall_objects PRIVATE ${FLYBALL_COMMON_DEFINES})
target_link_libraries(flyBall_objects PRIVATE ${FLYBALL_COMMON_LINKER_FLAGS})
target_include_directories(flyBall_objects PUBLIC $<BUILD_INTERFACE:${START_DIR}>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>>
    $<BUILD_INTERFACE:${START_DIR}/flyBall_ert_rtw>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/flyBall_ert_rtw>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/extern/include>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/simulink/include>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/rtw/c/src>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/rtw/c/src/ext_mode/common>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/rtw/c/ert>)
target_link_libraries(flyBall_objects PUBLIC $<$<BOOL:${FOUND_LIBM}>:m>)

add_executable(flyBall  ${START_DIR}/flyBall_ert_rtw/ert_main.cpp)
set_target_properties(flyBall PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(flyBall PUBLIC cxx_std_11)
target_compile_definitions(flyBall PRIVATE ${FLYBALL_COMMON_DEFINES})
target_link_libraries(flyBall PRIVATE ${FLYBALL_COMMON_LINKER_FLAGS})
target_link_libraries(flyBall PRIVATE flyBall_objects)
if(WIN32)
   target_link_libraries(flyBall PRIVATE kernel32
       ws2_32
       mswsock
       advapi32)
endif()

######################################################################
## Extract debug symbols
######################################################################
if(APPLE AND (CMAKE_C_COMPILER_ID STREQUAL "AppleClang"))
    string(APPEND _dsymutilcmd "$<IF:$<CONFIG:DEBUG,RELWITHDEBINFO>,"
                "xcrun;dsymutil;$<TARGET_FILE:flyBall>;--flat,"
                "true"
        >)
    add_custom_command(TARGET flyBall
            POST_BUILD
            COMMAND "${_dsymutilcmd}"
            COMMAND_EXPAND_LISTS
            BYPRODUCTS flyBall.dwarf)
endif()


######################################################################
## Target install and export rules
######################################################################
install(TARGETS flyBall_objects EXPORT flyBallTargets OBJECTS DESTINATION flyBall_ert_rtw)
install(TARGETS flyBall RUNTIME DESTINATION . OPTIONAL)
if(APPLE AND (CMAKE_C_COMPILER_ID STREQUAL "AppleClang"))
    install(FILES $<TARGET_FILE_DIR:flyBall>/flyBall.dwarf DESTINATION . OPTIONAL CONFIGURATIONS Debug RelWithDebInfo)
endif()
if(CMAKE_C_LINKER_SUPPORTS_PDB)
    install(FILES $<TARGET_PDB_FILE:flyBall> DESTINATION . OPTIONAL CONFIGURATIONS Debug RelWithDebInfo)
endif()
install(EXPORT flyBallTargets NAMESPACE flyBall:: DESTINATION flyBall_ert_rtw/export)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/flyBall.cmake 
 "include(\"\${CMAKE_CURRENT_LIST_DIR}/flyBallTargets.cmake\")" \n
    "target_include_directories(flyBall::flyBall_objects INTERFACE \n"
    "    \${MATLAB_ROOT}/extern/include" \n 
    "    \${MATLAB_ROOT}/simulink/include" \n 
    "    \${MATLAB_ROOT}/rtw/c/src" \n 
    "    \${MATLAB_ROOT}/rtw/c/src/ext_mode/common" \n 
    "    \${MATLAB_ROOT}/rtw/c/ert)" \n )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/flyBall.cmake DESTINATION flyBall_ert_rtw/export)

######################################################################
## Build success message
######################################################################
add_custom_command(TARGET flyBall POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "\\#\\#\\# Created executable: $<TARGET_FILE:flyBall>")

######################################################################
## Call toolchain hook function if defined
######################################################################
if(COMMAND toolchain_target_hook)
    toolchain_target_hook(flyBall)
endif()
