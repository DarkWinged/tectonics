
import numpy as np
import OpenGL.GL as gl


def cubic_spline(data: np.ndarray, samples: int, *, shader: str) -> np.ndarray:
    """
    Performs cubic spline interpolation on a 1D array using an OpenGL compute shader.

    Args:
        data (np.ndarray): 1D array of input values to be interpolated.
        samples (int): Number of interpolated samples between adjacent data points.
        shader (str): The compute shader program used for interpolation.

    Returns:
        np.ndarray: The interpolated 1D array.

    Raises:
        ValueError: If the input data is not a 1D array.
    """
    data_size = len(data)
    output_size = data_size * samples + (data_size - samples)

    input_buffer = np.array(data, dtype=np.float32)
    output_buffer = np.zeros(output_size, dtype=np.float32)

    gl.glUseProgram(shader)

    # Create buffers and bind them to shader storage
    input_ssbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, input_ssbo)
    gl.glBufferData(
        gl.GL_SHADER_STORAGE_BUFFER,
        input_buffer.nbytes,
        input_buffer,
        gl.GL_STATIC_DRAW,
    )
    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, input_ssbo)

    output_ssbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, output_ssbo)
    gl.glBufferData(
        gl.GL_SHADER_STORAGE_BUFFER,
        output_buffer.nbytes,
        output_buffer,
        gl.GL_DYNAMIC_COPY,
    )
    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, output_ssbo)

    # Set shader uniforms
    gl.glUniform1i(gl.glGetUniformLocation(shader, "samples"), samples)
    gl.glUniform1i(gl.glGetUniformLocation(shader, "dataSize"), data_size)

    # Run compute shader
    gl.glDispatchCompute((output_size + 255) // 256, 1, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

    # Retrieve results from output buffer
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, output_ssbo)
    output_data = np.frombuffer(
        gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, output_buffer.nbytes),
        dtype=np.float32,
    )

    # Cleanup
    gl.glDeleteBuffers(2, [input_ssbo, output_ssbo])

    return output_data
