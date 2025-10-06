#version 460

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuffer {
    float inputData[];
};

layout(std430, binding = 1) buffer OutputBuffer {
    float outputData[];
};

uniform int samples;  // Number of subdivisions between each point
uniform int dataSize; // Number of input data points

float cubicInterpolation(float prevPoint, float startPoint, float endPoint, float nextPoint, float t) {
    float a = (-prevPoint + 3.0 * startPoint - 3.0 * endPoint + nextPoint) * 0.5;
    float b = (2.0 * prevPoint - 5.0 * startPoint + 4.0 * endPoint - nextPoint) * 0.5;
    float c = (-prevPoint + endPoint) * 0.5;
    return ((a * t + b) * t + c) * t + startPoint;
}

void main() {
    uint globalIndex  = gl_GlobalInvocationID.x;
    
    int totalSize = (dataSize - 1) * (samples + 1) + 1;
    if (globalIndex  >= totalSize) return; // Avoid out-of-bounds writes
    
    int basePointIndex  = int(globalIndex  / (samples + 1)); // Index of the base data point
    int interpolationIndex = int(globalIndex  % (samples + 1)); // Position in interpolation range

    if (interpolationIndex == 0) {
        // Directly copy the original data points
        outputData[globalIndex ] = inputData[basePointIndex ];
    } else {
        // Interpolation
        float t = float(interpolationIndex) / float(samples + 1);

        float p0 = inputData[max(basePointIndex  - 1, 0)];
        float p1 = inputData[basePointIndex ];
        float p2 = inputData[min(basePointIndex  + 1, dataSize - 1)];
        float p3 = inputData[min(basePointIndex  + 2, dataSize - 1)];

        outputData[globalIndex ] = cubicInterpolation(p0, p1, p2, p3, t);
    }
}
