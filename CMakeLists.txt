cmake_minimum_required(VERSION 3.10)

# pytorch path
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")

add_subdirectory(external/uatlib)
add_subdirectory(external/CLI11)
# add_subdirectory(external/libtorch)
find_package(Torch REQUIRED)

add_executable(uatsim
  src/main.cpp
  src/airspace2d.cpp
  src/astar.cpp
  src/naive.cpp
  src/smart.cpp
  src/network.cpp
  )

target_include_directories(uatsim PRIVATE include)
target_link_libraries(uatsim PRIVATE uat cool jules fmt CLI11::CLI11 ${TORCH_LIBRARIES})
target_compile_features(uatsim PRIVATE cxx_std_20)
target_include_directories(uatsim PRIVATE include ${TORCH_INCLUDE_DIRS})

set_target_properties(uatsim PROPERTIES CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(uatsim PRIVATE "$<$<CONFIG:DEBUG>:-O0>")
target_compile_options(uatsim PRIVATE "$<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>")
target_compile_options(uatsim PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=address,leak,undefined>")
set_target_properties(uatsim PROPERTIES LINK_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,leak,undefined -fuse-ld=gold")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-parameter")

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    target_compile_options(uatsim PRIVATE "/W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(uatsim PRIVATE -Wall -Wextra -Werror -pedantic -Wno-missing-field-initializers)
endif()
