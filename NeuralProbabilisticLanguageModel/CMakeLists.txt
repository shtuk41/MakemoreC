set(TORCH_DLL_DIR "${Torch_DIR}/../../../lib")
file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set (MY_SOURCES
	"${CMAKE_SOURCE_DIR}/NeuralProbabilisticLanguageModel/source/main.cpp"
)

set(MY_HEADERS
    "${CMAKE_SOURCE_DIR}/include/MakemoreC.h"
)

source_group("Header Files" FILES ${MY_HEADERS})

add_executable(NeuralProbabilisticLanguageModel 
	${MY_SOURCES} 
)

target_sources(NeuralProbabilisticLanguageModel PRIVATE 
	${MY_SOURCES} 
    ${MY_HEADERS}
)

target_include_directories(NeuralProbabilisticLanguageModel PRIVATE 
	${CMAKE_SOURCE_DIR}/include
	${TORCH_INCLUDE_DIRS}
)

target_link_libraries(NeuralProbabilisticLanguageModel PRIVATE 
	"${TORCH_LIBRARIES}"
)

target_compile_features(NeuralProbabilisticLanguageModel PRIVATE cxx_std_20)

add_custom_command(TARGET NeuralProbabilisticLanguageModel POST_BUILD

	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_SOURCE_DIR}/build/NeuralProbabilisticLanguageModel/names.txt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/names.txt"
	
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_SOURCE_DIR}/build/NeuralProbabilisticLanguageModel/names1000.txt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/names1000.txt"

	

    # Copy shaders directory to both locations
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} "${CMAKE_BINARY_DIR}/NeuralProbabilisticLanguageModel"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} "${CMAKE_BINARY_DIR}/x64/$<CONFIG>"
    COMMENT "Deploying runtime DLL dependencies relative to Torch_DIR..."
)


set_target_properties(NeuralProbabilisticLanguageModel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>"
    #RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/bin"
    #RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build/bin"
	CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)




