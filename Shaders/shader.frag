#version 460

layout(location = 0) in vec3 pixelColor;
layout(location = 1) in vec2 pixelUV;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() 
{
    outColor = pixelColor.rgbr * texture(texSampler, pixelUV * 2.0f);
}