#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer InterpolatedData {
    float interpolatedGrid[];
};

layout(rgba32f, binding = 1) uniform image2D outputTexture;

uniform int gridWidth;
uniform int gridHeight;

int index(int x, int y, int width) {
    return y * width + x;  // Converts (x, y) to 1D index
}

void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);

    if (x >= gridWidth || y >= gridHeight) return;

    float value = interpolatedGrid[index(x, y, gridWidth)];
    value = clamp(value / 256.0, 0.0, 1.0);

    vec4 color = vec4(value, value, value, 1.0);
    imageStore(outputTexture, ivec2(x, y), color);
}
