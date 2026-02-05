cmake_minimum_required(VERSION 3.10)
project(LADDER VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找源文件
aux_source_directory(src/core CORE_SOURCES)
aux_source_directory(src/backend BACKEND_SOURCES)
aux_source_directory(src/compiler COMPILER_SOURCES)
aux_source_directory(src/utils UTIL_SOURCES)

# 创建静态库
add_library(ladder ${CORE_SOURCES} ${BACKEND_SOURCES} ${COMPILER_SOURCES} ${UTIL_SOURCES})
# 设置头文件路径（在库创建之后）
target_include_directories(ladder PUBLIC ${PROJECT_SOURCE_DIR}/include)

# 设置测试目标
enable_testing()
add_executable(test_ttype test/test_ttype.cpp)
add_executable(test_ttile test/test_ttile.cpp)
add_executable(test_scheduler test/test_scheduler.cpp)
add_executable(test_gemm test/test_gemm.cpp)

# 链接库
target_link_libraries(test_ttype PRIVATE ladder)
target_link_libraries(test_ttile PRIVATE ladder)
target_link_libraries(test_scheduler PRIVATE ladder)
target_link_libraries(test_gemm PRIVATE ladder)

# 添加测试
add_test(NAME ttype COMMAND test_ttype)
add_test(NAME ttile COMMAND test_ttile)
add_test(NAME scheduler COMMAND test_scheduler)
add_test(NAME gemm COMMAND test_gemm)
