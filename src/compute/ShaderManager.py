import functools
import glfw
import OpenGL.GL as gl
import numpy as np

from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class ShaderManager:
    """
    Manages OpenGL compute shaders and provides utility methods for their compilation and execution.

    Attributes:
        __shaders__ (dict): Stores compiled shader programs by name.
        __ogl_active__ (bool): Indicates whether an OpenGL context is currently active.
    """

    __shaders__: dict = {}
    __ogl_active__: bool = False

    def __init__(self) -> None:
        if not self.__ogl_active__:
            self.__activate_ogl__()

    @classmethod
    def __activate_ogl__(cls) -> None:
        """
        Initializes an OpenGL context using GLFW if one is not already active.

        Raises:
            RuntimeError: If GLFW initialization or window creation fails.
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
        Compiles a compute shader from source and registers it under a given name.

        Args:
            shader_source (str): Source code for the compute shader.
            shader_name (str): Name used to register the compiled shader program.

        Raises:
            RuntimeError: If shader compilation or linking fails.
        """
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        gl.glShaderSource(shader, shader_source)
        gl.glCompileShader(shader)

        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            error = gl.glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Compute shader compilation failed: {error}")

        program = gl.glCreateProgram()
        gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)

        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            error = gl.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Compute shader linking failed: {error}")

        cls.__shaders__[shader_name] = program

    @classmethod
    def shader_context(cls, shader_name: str, shader_path: str) -> Callable[P, T]:
        """
        Decorator that compiles a compute shader if necessary and associates it with a function.

        The decorated function will automatically use the specified shader when called.

        Args:
            shader_name (str): Name used to register or retrieve the compiled shader program.
            shader_path (str): File path to the compute shader source code.

        Returns:
            Callable[P, T]: A decorator that wraps the target function.
        """
        if not cls.__ogl_active__:
            cls.__activate_ogl__()
        if shader_name not in cls.__shaders__:
            with open(shader_path) as file:
                cls.__compile_compute_shader__(file.read(), shader_name)
        shader = cls.__shaders__[shader_name]

        def wrapper(func: Callable[P, T]) -> Callable[P, T]:
            """
            Wraps a function so that it uses the specified compute shader when invoked.

            Args:
                func (Callable[P, T]): A function that accepts 'shader' as a keyword argument.

            Returns:
                Callable[P, T]: The wrapped classmethod.
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
        Performs 2D cubic spline interpolation on a grid, processing rows and then columns.

        Args:
            grid (np.ndarray): 2D array representing the input grid.
            samples (int): Number of interpolated samples between adjacent data points.

        Returns:
            np.ndarray: A 2D array with the interpolated result.
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
        Performs 2D cubic spline interpolation on a grid in smaller chunks to reduce memory usage.

        The grid is divided row-wise and column-wise into subregions, each of which is interpolated
        independently via the 'cubic_spline_2d' method.

        Args:
            grid (np.ndarray): 2D array representing the input grid.
            samples (int): Number of interpolated samples between adjacent data points.
            chunk_size (int, optional): Size of each processing chunk. Defaults to 256.

        Returns:
            np.ndarray: A 2D array containing the interpolated result.
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
