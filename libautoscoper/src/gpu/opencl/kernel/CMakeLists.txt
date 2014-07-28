SET(SHADER_TO_HEADER "shader_to_header")

set(opencl_shaders
	gpu/opencl/kernel/Compositor.cl
	gpu/opencl/kernel/ContrastFilter.cl
	gpu/opencl/kernel/GaussianFilter.cl
	gpu/opencl/kernel/Ncc.cl
	gpu/opencl/kernel/NccSum.cl
	gpu/opencl/kernel/RadRenderer.cl
	gpu/opencl/kernel/RayCaster.cl
	gpu/opencl/kernel/SharpenFilter.cl
	gpu/opencl/kernel/SobelFilter.cl
)

foreach(SHADER_FILENAME ${opencl_shaders})

    # Get the shader filename without the extension
    get_filename_component(SHADER_NAME ${SHADER_FILENAME} NAME_WE)

    # Construct the header file full path
    set(SHADER_NAME "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.h")

    list(APPEND GENERATED_LIST ${SHADER_NAME})

    add_custom_command(OUTPUT  ${SHADER_NAME}
                       COMMAND cmake -DSHADER_FILENAME=${SHADER_FILENAME}
                                     -DOUTPUT_FOLDER=${CMAKE_CURRENT_BINARY_DIR}/gpu/opencl/kernel
                                     -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu/opencl/kernel/cmake/shader_to_header.cmake
                       DEPENDS src/${SHADER_FILENAME}
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

add_custom_target(${SHADER_TO_HEADER} DEPENDS ${GENERATED_LIST})

SET(opencl_shaders_src ${GENERATED_LIST} PARENT_SCOPE)

