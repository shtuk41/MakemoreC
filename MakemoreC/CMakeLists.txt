find_package(GTest CONFIG REQUIRED)

set (MY_SOURCES
	"${CMAKE_SOURCE_DIR}/MakemoreC/source/test.cpp"
	"${CMAKE_SOURCE_DIR}/MakemoreC/source/trace.cpp"
	"${CMAKE_SOURCE_DIR}/MakemoreC/source/value.cpp"
)

set(MY_HEADERS
    "${CMAKE_SOURCE_DIR}/MakemoreC/include/functional.h"
    "${CMAKE_SOURCE_DIR}/MakemoreC/include/neuron.h"
	"${CMAKE_SOURCE_DIR}/MakemoreC/include/trace.h"
	"${CMAKE_SOURCE_DIR}/MakemoreC/include/value.h"
	"${CMAKE_SOURCE_DIR}/include/MakemoreC.h"
)

source_group("Header Files" FILES ${MY_HEADERS})

add_executable(MakemoreC 
	${MY_SOURCES} 
)

target_sources(MakemoreC PRIVATE 
	${MY_SOURCES} 
    ${MY_HEADERS}
)

target_include_directories(MakemoreC PRIVATE 
	${CMAKE_SOURCE_DIR}/MakemoreC/include
	${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MakemoreC PRIVATE 
	 GTest::gtest_main   # Automatically provides a standard main() entry point
     GTest::gtest        # The core GoogleTest library
)

target_compile_features(MakemoreC PRIVATE cxx_std_20)

add_custom_command(TARGET MakemoreC POST_BUILD

	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_SOURCE_DIR}/build/MakemoreC/names.txt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/names.txt"
	
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_SOURCE_DIR}/build/MakemoreC/names1000.txt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/names1000.txt"

	
	#COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/notdefinedfilename.png" "${CMAKE_SOURCE_DIR}/build/notdefinedfilename.png"
    #COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/notdefinedfilename.png" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/notdefinedfilename.png"
)


set_target_properties(MakemoreC PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>"
    #RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/bin"
    #RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build/bin"
	CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_test(NAME MakemoreC COMMAND MakemoreTest)




