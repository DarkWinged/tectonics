#version 430

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputData {
    float inputData[];
};

layout(std430, binding = 1) buffer OutputData {
    float outputData[];
};

layout(std430, binding = 2) buffer CurvatureBuffer {
    float curvatureConstraints[];
};

layout(std430, binding = 3) buffer SecondDerivativeBuffer {
    float secondDerivatives[];
};

uniform int samples;
uniform int dataSize;

void compute_second_derivatives() {
    float diagonalTerms[256];
    float offDiagonalTerms[256];
    
    secondDerivatives[0] = 0.0;
    diagonalTerms[0] = 1.0;
    offDiagonalTerms[0] = 0.0;
    
    for (int i = 1; i < dataSize - 1; i++) {
        diagonalTerms[i] = 4.0 - offDiagonalTerms[i - 1];
        offDiagonalTerms[i] = 1.0 / diagonalTerms[i];
        secondDerivatives[i] = (curvatureConstraints[i] - secondDerivatives[i - 1]) / diagonalTerms[i];
    }
    
    secondDerivatives[dataSize - 1] = 0.0;
    
    for (int i = dataSize - 2; i >= 0; i--) {
        secondDerivatives[i] -= offDiagonalTerms[i] * secondDerivatives[i + 1];
    }
}

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= (dataSize - 1) * samples + dataSize) return;

    int segment = i / (samples + 1);
    float t = float(i % (samples + 1)) / float(samples + 1);
    
    float y0 = inputData[max(0, segment - 1)];
    float y1 = inputData[segment];
    float y2 = inputData[min(dataSize - 1, segment + 1)];
    float y3 = inputData[min(dataSize - 1, segment + 2)];

    if (segment < dataSize - 1) {
        curvatureConstraints[segment] = 3.0 * (y2 - y1) - 3.0 * (y1 - y0);
    }
    
    barrier();
    compute_second_derivatives();

    float cubic = (secondDerivatives[segment + 1] - secondDerivatives[segment]) / 6.0;
    float quadratic = secondDerivatives[segment] / 2.0;
    float linear = (y2 - y1) - (secondDerivatives[segment + 1] + 2.0 * secondDerivatives[segment]) / 6.0;
    float constant = y1;

    outputData[i] = constant + linear * t + quadratic * t * t + cubic * t * t * t;
    
    if (i == (dataSize - 1) * samples + dataSize - 1) {
        outputData[i] = inputData[dataSize - 1];
    }
}
