import functools
import os
import glfw
import OpenGL.GL as gl
import numpy as np

from PIL import Image
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class ShaderManager:
    """
    A static class for managing OpenGL compute shaders and running them with registered controller functions.

    Attributes:
        __shaders__ (dict): A dictionary of registered shader programs.
        __ogl_active__ (bool): A flag indicating whether an OpenGL context is active.

    Notes:
        New compute shaders can be registered by decorating a function with the shader_context decorator.
        Registered functions can be accessed as class methods and automatically run the associated shader program.
    """

    __shaders__: dict = {}
    __ogl_active__: bool = False

    def __init__(self) -> None:
        if not self.__ogl_active__:
            self.__activate_ogl__()

    @classmethod
    def __activate_ogl__(cls) -> None:
        """
        Initializes an OpenGL context using GLFW. Sets ogl activivation status to True via the __ogl_active__ class attribute.

        Raises:
            RuntimeError: If the initialization of the OpenGL context fails.

        Notes:
            The method is called by the class constructor and shader_context decorator to ensure an OpenGL context is active.
        """
        print("Initializing OpenGL context...")

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(1, 1, "Hidden", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create an OpenGL context")

        glfw.make_context_current(window)

        print("OpenGL Context Initialized!")
        print("OpenGL Version:", gl.glGetString(gl.GL_VERSION).decode())
        cls.__ogl_active__ = True

    @classmethod
    def __compile_compute_shader__(cls, shader_source: str, shader_name: str) -> None:
        """
        Compiles a compute shader and links it to a program.

        Args:
            shader_source (str): The source code of the compute shader.
            shader_name (str): The name of the shader to register.

        Raises:
            RuntimeError: If the shader compilation or linking

        Notes:
            The compiled shader program is registered to the class in the __shaders__ dictionary.
            Utilized by the shader_context decorator to compile and register shaders.
        """
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        gl.glShaderSource(shader, shader_source)
        gl.glCompileShader(shader)

        # Check for errors
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            error = gl.glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Compute shader compilation failed: {error}")

        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)

        # Check linking errors
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            error = gl.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Compute shader linking failed: {error}")

        cls.__shaders__[shader_name] = program

    # helper decorator for managing shader wrappers functions
    @classmethod
    def shader_context(cls, shader_name: str, shader_path: str) -> Callable[P, T]:
        """
        Decorator for registering a compute shader and a function to run it to the ShaderManager class.

        Args:
            shader_name (str): The name of the shader to register.
            shader_path (str): The path to the shader source file.

        Returns:
            callable: The wrapper function.
        """
        if not cls.__ogl_active__:
            cls.__activate_ogl__()
        if shader_name not in cls.__shaders__:
            with open(shader_path) as file:
                cls.__compile_compute_shader__(file.read(), shader_name)
        shader = cls.__shaders__[shader_name]

        def wrapper(func: Callable[P, T]) -> Callable[P, T]:
            """
            Wrapper function to register the shader and function to the class.

            Args:
                func (Callable[P, T]): A shader processing function, which accepts the shader program as a keyword argument.

            Returns:
                Callable[P, T]: The Decorated function.

            Notes:
                Utilizes the shader variable from the outer scope to pass the callable function as a keyword argument.
                Wraped functions are required to accept the shader keyword argument in order to receive the shader program.
            """

            @classmethod
            @functools.wraps(func)
            def decorator(cls, *args: P.args, **kwargs: P.kwargs) -> T:
                gl.glUseProgram(shader)
                return func(*args, **kwargs, shader=shader)

            setattr(cls, func.__name__, decorator)
            return decorator

        return wrapper

    @classmethod
    def cubic_spline_2d(cls, grid: np.ndarray, samples: int) -> np.ndarray:
        """
        Runs a 2D cubic spline interpolation using the compute shader row-wise and then column-wise.

        Args:
            grid (np.ndarray): The input grid to interpolate.
            samples (int): The number of samples to interpolate between each data point.

        Returns:
            np.ndarray: The interpolated grid.

        Notes:
            Utilizes the registered cubic_spline function to interpolate the grid.
        """
        grid_height, grid_width = grid.shape
        output_width = grid_width * samples + (grid_width - samples)
        output_height = grid_height * samples + (grid_height - samples)

        row_interpolated = np.zeros((grid_height, output_width), dtype=np.float32)
        final_output = np.zeros((output_height, output_width), dtype=np.float32)

        for row_idx in range(grid_height):
            row_interpolated[row_idx] = cls.cubic_spline(grid[row_idx], samples)

        for col_idx in range(output_width):
            final_output[:, col_idx] = cls.cubic_spline(
                row_interpolated[:, col_idx], samples
            )

        return final_output

    @classmethod
    def cubic_spline_2d_chunked(
        cls, grid: np.ndarray, samples: int, chunk_size: int = 256
    ) -> np.ndarray:
        """
        Runs a 2D cubic spline interpolation using the compute shader row-wise and then column-wise in chunks.

        Args:
            grid (np.ndarray): The input grid to interpolate.
            samples (int): The number of samples to interpolate between each data point.
            chunk_size (int, optional): The size of the chunks to interpolate. Defaults to 256.

        Returns:
            np.ndarray: The interpolated grid.

        Notes:
            Utilizes the cubic_spline_2d method to interpolate chunks of the grid.
        """
        grid_height, grid_width = grid.shape
        output_width = grid_width * samples + (grid_width - samples)
        output_height = grid_height * samples + (grid_height - samples)

        final_output = np.zeros((output_height, output_width), dtype=np.float32)

        for y_idx, y in enumerate(range(0, grid_height, chunk_size)):
            for x_idx, x in enumerate(range(0, grid_width, chunk_size)):
                y_start = max(0, y - y_idx)
                x_start = max(0, x - x_idx)
                y_end = min(y_start + chunk_size + y_idx, grid_height)
                x_end = min(x_start + chunk_size + x_idx, grid_width)

                chunk = grid[y_start:y_end, x_start:x_end]
                interpolated_chunk = cls.cubic_spline_2d(chunk, samples)

                y_out_start = y_start * samples + y_start
                x_out_start = x_start * samples + x_start
                y_out_end = min(
                    y_out_start + interpolated_chunk.shape[0], output_height
                )
                x_out_end = min(x_out_start + interpolated_chunk.shape[1], output_width)

                cropped_chunk = interpolated_chunk[
                    : y_out_end - y_out_start, : x_out_end - x_out_start
                ]

                final_output[y_out_start:y_out_end, x_out_start:x_out_end] = (
                    cropped_chunk
                )
        return final_output


@ShaderManager.shader_context(
    "grid_to_tex", os.path.realpath("src/compute/shaders/grid_to_tex.glsl")
)
def write_grid_to_texture(
    interpolated_grid: np.ndarray, /, shader: str, output_filename="image.png"
):
    """
    Writes an interpolated grid to a texture using an OpenGL compute shader and saves it as a PNG image.
    Args:
        interpolated_grid (np.ndarray): The interpolated grid data as a 2D NumPy array.
        shader (str): The shader program to use for writing the grid to the texture.
        output_filename (str, optional): The filename for the output PNG image. Defaults to "output.png".
    Raises:
        ValueError: If the interpolated grid is not a 2D NumPy array.
    Notes:
        Registered as a class method to the ShaderManager class via the ShaderManager.shader_context decorator.
    """
    interpolated_grid = np.array(interpolated_grid, dtype=np.float32)
    grid_height, grid_width = interpolated_grid.shape

    # Create OpenGL texture
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexStorage2D(gl.GL_TEXTURE_2D, 1, gl.GL_RGBA32F, grid_width, grid_height)

    # Bind texture to shader
    gl.glBindImageTexture(
        1, texture, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RGBA32F
    )

    # Create OpenGL buffer for interpolated data
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

    # Dispatch Compute Shader
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
    realPath = os.path.realpath(output_filename)
    Image.fromarray(output_image).save(realPath)

    # Cleanup
    gl.glDeleteTextures(1, [texture])
    gl.glDeleteBuffers(1, [buffer])

    print(f"Saved interpolated grid to {realPath}")


@ShaderManager.shader_context(
    "cubic_spline", os.path.realpath("src/compute/shaders/cubic_spline.glsl")
)
def cubic_spline(data: np.ndarray, samples: int, *, shader: str) -> np.ndarray:
    """
    Calculates the cubic spline interpolation of a 1D array using an OpenGL compute shader.
    Args:
        data (np.ndarray): The input data to interpolate.
        samples (int): The number of samples to interpolate between each data point.
        shader (str): The shader program to use for cubic spline interpolation.
    Returns:
        np.ndarray: The interpolated data.
    Raises:
        ValueError: If the input data is not a 1D NumPy array.
    Notes:
        Registered as a class method to the ShaderManager class via the ShaderManager.shader_context decorator.
    """
    data_size = len(data)
    output_size = data_size * samples + (data_size - samples)

    input_buffer = np.array(data, dtype=np.float32)
    output_buffer = np.zeros(output_size, dtype=np.float32)

    gl.glUseProgram(shader)

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

    gl.glUniform1i(gl.glGetUniformLocation(shader, "samples"), samples)
    gl.glUniform1i(gl.glGetUniformLocation(shader, "dataSize"), data_size)

    gl.glDispatchCompute((output_size + 255) // 256, 1, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, output_ssbo)
    output_data = np.frombuffer(
        gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, output_buffer.nbytes),
        dtype=np.float32,
    )

    gl.glDeleteBuffers(2, [input_ssbo, output_ssbo])
    return output_data
