#version 460

layout(binding = 0) uniform UniformBufferObject 
{
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;


layout(location = 0) out vec3 pixelColor;
layout(location = 1) out vec2 pixelUV;

void main() 
{
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0f);
    pixelColor = inColor;
	pixelUV = inUV;
}