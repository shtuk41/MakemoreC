set(TORCH_DLL_DIR "${Torch_DIR}/../../../lib")
file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.so")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set (MY_SOURCES
	"${CMAKE_SOURCE_DIR}/MakemoreCLibTorch/source/main.cpp"
)

set(MY_HEADERS
    "${CMAKE_SOURCE_DIR}/include/MakemoreC.h"
)

source_group("Header Files" FILES ${MY_HEADERS})

add_executable(MakemoreCLibTorch 
	${MY_SOURCES} 
)

target_sources(MakemoreCLibTorch PRIVATE 
	${MY_SOURCES} 
    ${MY_HEADERS}
)

target_include_directories(MakemoreCLibTorch PRIVATE 
	${CMAKE_SOURCE_DIR}/include
	${TORCH_INCLUDE_DIRS}
)

target_link_libraries(MakemoreCLibTorch PRIVATE 
	${TORCH_LIBRARIES}
)

target_compile_features(MakemoreCLibTorch PRIVATE cxx_std_20)

if(WIN32)
    add_custom_command(TARGET MakemoreCLibTorch POST_BUILD

	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_SOURCE_DIR}/build/MakemoreCLibTorch/names.txt"
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/names.txt"
	
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_SOURCE_DIR}/build/MakemoreCLibTorch/names1000.txt"
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>/names1000.txt"


    # Copy shaders directory to both locations
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} "${CMAKE_BINARY_DIR}/MakemoreCLibTorch"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} "${CMAKE_BINARY_DIR}/x64/$<CONFIG>"
   	COMMENT "Deploying windows runtime DLL dependencies relative to Torch_DIR..."
)
elseif(LINUX)
    add_custom_command(TARGET MakemoreCLibTorch POST_BUILD

    	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names.txt" "${CMAKE_BINARY_DIR}/bin/names.txt"
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/items/names1000.txt" "${CMAKE_BINARY_DIR}/bin/names1000.txt"


    # Copy shaders directory to both locations
	#COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} "${CMAKE_BINARY_DIR}/bin"
    	COMMENT "Deploying linux  runtime SO dependencies relative to Torch_DIR..."
)
endif()




if(WIN32)
	target_compile_features(MicrogradC.Test PRIVATE cxx_std_20)

    set_target_properties(MakemoreCLibTorch PROPERTIES
    		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>"
    		#RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/bin"
    		#RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build/bin"
		CXX_STANDARD 20
    		CXX_STANDARD_REQUIRED ON
    	CXX_EXTENSIONS OFF
)

elseif(LINUX)
	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)
	
	set_target_properties(MakemoreCLibTorch PROPERTIES
	    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
	    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
	    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
	)
endif()





