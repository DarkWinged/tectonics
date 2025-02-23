from PIL import Image
import OpenGL.GL as gl
import numpy as np
import os

def write_grid_to_texture(
    interpolated_grid: np.ndarray, /, shader: str, output_filename="image.png"
):
    """
    Writes a 2D interpolated grid to a texture using an OpenGL compute shader, then saves it as a PNG image.

    Args:
        interpolated_grid (np.ndarray): The 2D grid data to be saved.
        shader (str): The compute shader program used to write data to the texture.
        output_filename (str, optional): Name of the output PNG file. Defaults to "image.png".

    Raises:
        ValueError: If the input grid is not 2D.
    """
    interpolated_grid = np.array(interpolated_grid, dtype=np.float32)
    grid_height, grid_width = interpolated_grid.shape

    # Create OpenGL texture
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexStorage2D(gl.GL_TEXTURE_2D, 1, gl.GL_RGBA32F, grid_width, grid_height)

    # Bind texture to the shader
    gl.glBindImageTexture(
        1, texture, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RGBA32F
    )

    # Create OpenGL buffer for the grid data
    interpolated_data = interpolated_grid.flatten()
    buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer)
    gl.glBufferData(
        gl.GL_SHADER_STORAGE_BUFFER,
        interpolated_data.nbytes,
        interpolated_data,
        gl.GL_STATIC_DRAW,
    )
    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, buffer)

    # Set shader uniforms
    grid_width_loc = gl.glGetUniformLocation(shader, "gridWidth")
    grid_height_loc = gl.glGetUniformLocation(shader, "gridHeight")
    gl.glUniform1i(grid_width_loc, grid_width)
    gl.glUniform1i(grid_height_loc, grid_height)

    # Dispatch the compute shader
    workgroup_size = 16
    gl.glDispatchCompute(
        (grid_width + workgroup_size - 1) // workgroup_size,
        (grid_height + workgroup_size - 1) // workgroup_size,
        1,
    )
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # Retrieve texture data
    output_data = np.empty((grid_height, grid_width, 4), dtype=np.float32)
    gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_FLOAT, output_data)

    # Convert to grayscale image and save as PNG
    output_image = (output_data[:, :, 0] * 255).astype(np.uint8)
    real_path = os.path.realpath(output_filename)
    Image.fromarray(output_image).save(real_path)

    # Cleanup
    gl.glDeleteTextures(1, [texture])
    gl.glDeleteBuffers(1, [buffer])

    print(f"Saved interpolated grid to {real_path}")