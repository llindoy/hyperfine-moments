cmake_minimum_required (VERSION 3.11)
project(hyperfine_moments LANGUAGES CXX)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

#various tags for specifying the features we want to use when compiling
option(USE_CUDA "Compile with cuda" OFF)
option(USE_OPENMP "Compile with openmp" ON)
option(USE_CHAI "Compile with Chaiscript" ON)
option(USE_CEREAL "Compile with cereal serialisation" OFF)

if(USE_OPENMP)
    add_compile_definitions(USE_OPENMP)
endif()

if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()

#setup the compiler options
include(CheckLanguage)
if(COMPILER_SET)
else()
    if(MSVC)
        message(FATAL_ERROR "Request MSVC compiler.  I don't know how this works so exiting.")
    else()
        set(CMAKE_CXX_FLAGS "-O3 -g -Wshadow -pedantic ${CMAKE_CXX_FLAGS}")
        option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" ON)
        if(WITH_WERROR)
            #set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
        endif(WITH_WERROR)
        
        option(CLANG_USE_LIBCPP "Use libc++ for clang compilation" OFF)

        #additional compilation flags for GNU compilers
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

            message(STATUS "${CMAKE_CXX_COMPILER_ID}")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fwhole-program -fcx-fortran-rules")
            if(USE_OPENMP)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            endif()
        #additional compilation flags for inteal compilers
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=3 -ipo -O3 -xHost -restrict")
            message(STATUS "using intel compiler.")
            if(USE_OPENMP)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
            endif()

        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -qopt-report=3 ")
            if(USE_OPENMP)
                message(STATUS "Using Openmp")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
            endif()

        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            if(CLANG_USE_LIBCPP)
                set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
            endif()
            if(USE_OPENMP)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            endif()
        else()
            message(STATUS "${CMAKE_CXX_COMPILER_ID}")
            message(FATAL_ERROR "Compiler not supported.")
        endif()
        

        if(CMAKE_VERSION VERSION_LESS 3.1)
            set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
        else()
          if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD STREQUAL "98")
            set(CMAKE_CXX_STANDARD 17)
          endif()

          if(CMAKE_CXX_STANDARD GREATER 17)
            cmake_minimum_required(VERSION 3.8)
          endif()

          set(CMAKE_CXX_STANDARD_REQUIRED ON)
        endif()
    endif()
    set(COMPILER_SET TRUE INTERNAL BOOL "")
endif()

#now add the source file to the executable
add_executable(${CMAKE_PROJECT_NAME} main.cpp)

#make sure we have the linalg library in the correct location and determine the version of blas and lapack to use

include(FetchContent)
#now make sure we have the rapidjson library 
#now we can also set up rapidjson
set(RAPIDJSON_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/rapidjson)
if(EXISTS ${RAPIDJSON_ROOT_DIR})
    SET(rapidjson_repo "file://${RAPIDJSON_ROOT_DIR}/")
else()
    SET(rapidjson_repo "https://github.com/Tencent/rapidjson")
endif()

FetchContent_Declare(rapidjson
    GIT_REPOSITORY ${rapidjson_repo}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL
)

FetchContent_Populate(rapidjson)

add_definitions(-DRAPIDJSON_LIBRARY_FOUND)
set(EOS_INCLUDE_DIRECTORIES "${rapidjson_SOURCE_DIR}/include/;${EOS_INCLUDE_DIRECTORIES}" CACHE INTERNAL "")


if(USE_CHAI)
    SET(CHAISCRIPT_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/ChaiScript)
    SET(CHAISCRIPT_EXTRA_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/ChaiScript_Extras)
    if(EXISTS ${CHAISCRIPT_ROOT_DIR})
        SET(chaiscript_repo "file://${CHAISCRIPT_ROOT_DIR}/")
    else()
        SET(chaiscript_repo "https://github.com/ChaiScript/ChaiScript.git")
    endif()
    message(STATUS "Chai repo ${chaiscript_repo}")

    if(EXISTS ${CHAISCRIPT_EXTRAS_ROOT_DIR})
        SET(chaiscript_repo "file://${CHAISCRIPT_EXTRAS_ROOT_DIR}/")
    else()
        SET(chaiscript_extras_repo "https://github.com/ChaiScript/ChaiScript_Extras.git")
    endif()
    
    FetchContent_Declare(chaiscript 
        GIT_REPOSITORY ${chaiscript_repo}
        GIT_TAG "develop"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL
    )

    FetchContent_Populate(chaiscript)


    include(FetchContent)
    FetchContent_Declare(chaiscript_extras
        GIT_REPOSITORY ${chaiscript_extras_repo}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL
    )

    FetchContent_Populate(chaiscript_extras)

    add_definitions(-DUSE_CHAISCRIPT)
    set(EOS_INCLUDE_DIRECTORIES "${chaiscript_extras_SOURCE_DIR}/include/;${EOS_INCLUDE_DIRECTORIES}" CACHE INTERNAL "")
    set(EOS_INCLUDE_DIRECTORIES "${chaiscript_SOURCE_DIR}/include/;${EOS_INCLUDE_DIRECTORIES}" CACHE INTERNAL "")

    message(STATUS "chaiscript source directory ${chaiscript_SOURCE_DIR}")
endif()


if(USE_CEREAL)
    set(CEREAL_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/cereal)
    if(EXISTS ${CEREAL_ROOT_DIR})
        SET(cereal_repo "file://${CEREAL_ROOT_DIR}/")
    else()
        SET(cereal_repo "https://github.com/USCiLab/cereal.git")
    endif()

    FetchContent_Declare(cereal
        GIT_REPOSITORY ${cereal_repo}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL
    )

    FetchContent_Populate(cereal)

    add_definitions(-DCEREAL_LIBRARY_FOUND)
    set(EOS_INCLUDE_DIRECTORIES "${cereal_SOURCE_DIR}/include/;${EOS_INCLUDE_DIRECTORIES}" CACHE INTERNAL "")
endif()


if(DEFINED BLA_VENDOR)
    if(BLA_VENDOR MATCHES "Intel")
        find_package(LAPACK REQUIRED)
        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            set(MKL_INCLUDE_DIRS ${MKLROOT}/include)
            #FIND_PATH(MKL_INCLUDE_DIR "mkl_cblas.h" PATHS ${MKLROOT}/include)
            message(STATUS ${MKL_INCLUDE_DIRS})
            add_compile_definitions(USE_MKL)
            add_compile_definitions(BLAS_HEADER_INCLUDED)
            add_compile_definitions(NOT_ALLOWS_RETURN_TYPE )
            add_compile_definitions(BLAS_NO_TRAILING_UNDERSCORE )
            if(DLA_VENDOR MATCHES "64ilp")
                add_compile_definitons(BLAS_64_BIT)
            endif()
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
            set(USE_SPARSE_BLAS OFF)
        endif()
    elseif(BLA_VENDOR MATCHES "OpenBLAS")   
        add_compile_definitions(USE_OPENBLAS)
        set(USE_DEFAULT_FIND_BLAS ON)
    elseif(BLA_VENDOR MATCHES "FLAME")
        #here we allow for us to handle the multithreaded case.  This is not dealt with by the standard FindBLAS and FindLapack files.  
        #this has been setup to work with the aocl versions of blis and flame
        if(USE_OPENMP)
            #first we attempt to find the standard FLAME LIBRARY
            set(BLA_VENDOR "FLAME")
            include(FindBLAS)
            if(NOT BLAS_FOUND)
                message(FATAL_ERROR "BLAS NOT FOUND")
            endif()
            check_blas_libraries(
              BLAS_LIBRARIES
              BLAS
              sgemm
              ""
              "blis-mt"
              ""
              ""
              ""
              )
            message(STATUS ${BLAS_LIBRARIES})
            set(BLA_VENDOR "FLAME")
            find_package(LAPACK REQUIRED)
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            set(USE_DEFAULT_FIND_BLAS OFF)
        else()
            add_compile_definitions(USE_FLAME)
            set(USE_DEFAULT_FIND_BLAS ON)
        endif()
    else()
        set(USE_DEFAULT_FIND_BLAS ON)
    endif()
else()
    set(USE_DEFAULT_FIND_BLAS ON)
endif()

#if we aren't dealing with one of the vendors we treat specially then we will just go ahead and use the default
#findBLAS and findLAPACK calls
if(USE_DEFAULT_FIND_BLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    message(STATUS ${BLAS_LIBRARIES})
    set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${EOS_INCLUDE_DIRECTORIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LINALG_INCLUDE_DIRECTORIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${UTILS_INCLUDE_DIRECTORIES})
target_link_libraries(${CMAKE_PROJECT_NAME} utils linalg ${LINALG_LIBS} ${LIBS}) 
