import os
from .cublic_spline import cubic_spline
from .write_to_tex import write_grid_to_texture
from .ShaderManager import ShaderManager


ShaderManager.shader_context(
    "grid_to_tex", os.path.realpath("src/compute/shaders/grid_to_tex.glsl")
)(write_grid_to_texture)

ShaderManager.shader_context(
    "cubic_spline", os.path.realpath("src/compute/shaders/cubic_spline.glsl")
)(cubic_spline)
