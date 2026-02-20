cmake_minimum_required(VERSION 3.10)
project(LADDER VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LADDER_ENABLE_CUDA "Enable CUDA runtime support" OFF)

option(LADDER_ENABLE_MLIR "Enable MLIR dialect and passes" ON)

# 查找源文件
aux_source_directory(src/core CORE_SOURCES)
aux_source_directory(src/backend BACKEND_SOURCES)
aux_source_directory(src/compiler COMPILER_SOURCES)
aux_source_directory(src/utils UTIL_SOURCES)

# 创建静态库
add_library(ladder ${CORE_SOURCES} ${BACKEND_SOURCES} ${COMPILER_SOURCES} ${UTIL_SOURCES})
# 设置头文件路径（在库创建之后）
target_include_directories(ladder PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(LADDER_ENABLE_CUDA)
	find_package(CUDAToolkit QUIET)
	if(CUDAToolkit_FOUND)
		enable_language(CUDA)
		set(CMAKE_CUDA_STANDARD 17)
		target_compile_definitions(ladder PUBLIC LADDER_ENABLE_CUDA)
		target_link_libraries(ladder PUBLIC CUDA::cudart CUDA::cublas)
		target_sources(ladder PRIVATE ${PROJECT_SOURCE_DIR}/src/backend/cuda_gemm.cu)
	else()
		message(WARNING "CUDA toolkit not found; CUDA runtime support disabled.")
	endif()
endif()

if(LADDER_ENABLE_MLIR)
	if(NOT MLIR_DIR AND EXISTS "/hyz/project/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/cmake/mlir/MLIRConfig.cmake")
		set(MLIR_DIR "/hyz/project/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/cmake/mlir")
	endif()
	find_package(MLIR QUIET CONFIG)
	if(MLIR_FOUND)
		message(STATUS "Found MLIR: ${MLIR_DIR}")
		list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
		include(TableGen)
		include(AddMLIR)
		include(AddLLVM)
		add_subdirectory(mlir)

		add_mlir_tool(ladder-opt tools/ladder_opt.cpp)
		target_link_libraries(ladder-opt PRIVATE
			LadderTransforms
			LadderDialect
			MLIRFuncDialect
			MLIRArithDialect
			MLIRIR
			MLIRParser
			MLIRPass
			MLIRSupport
		)
		target_include_directories(ladder-opt PRIVATE
			${MLIR_INCLUDE_DIRS}
			${PROJECT_SOURCE_DIR}/include
			${PROJECT_BINARY_DIR}/include
			${PROJECT_BINARY_DIR}/mlir
		)
	else()
		message(WARNING "MLIR not found; ladder dialect/pass will not be built.")
	endif()
endif()

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

# example autotune runner
add_executable(run_autotune tools/run_autotune.cpp)
target_link_libraries(run_autotune PRIVATE ladder)

add_executable(run_ladder_ir tools/run_ladder_ir.cpp)
target_link_libraries(run_ladder_ir PRIVATE ladder)

# 添加测试
add_test(NAME ttype COMMAND test_ttype)
add_test(NAME ttile COMMAND test_ttile)
add_test(NAME scheduler COMMAND test_scheduler)
add_test(NAME gemm COMMAND test_gemm)
