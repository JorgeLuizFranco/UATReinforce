cmake_minimum_required(VERSION 3.10)

add_subdirectory(external/uatlib)
add_subdirectory(external/CLI11)

add_executable(uatsim
  src/main.cpp
  src/airspace3d.cpp
  src/astar.cpp
  src/naive.cpp
  src/smart.cpp)

target_include_directories(uatsim PRIVATE include)
target_link_libraries(uatsim PRIVATE uat cool jules fmt CLI11::CLI11)
target_compile_features(uatsim PRIVATE cxx_std_20)

set_target_properties(uatsim PROPERTIES CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(uatsim PRIVATE "$<$<CONFIG:DEBUG>:-O0>")
target_compile_options(uatsim PRIVATE "$<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>")
target_compile_options(uatsim PRIVATE "$<$<CONFIG:DEBUG>:-fsanitize=address,leak,undefined>")
set_target_properties(uatsim PROPERTIES LINK_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,leak,undefined -fuse-ld=gold")

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    target_compile_options(uatsim PRIVATE "/W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(uatsim PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
