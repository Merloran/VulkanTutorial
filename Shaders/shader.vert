#version 460

//layout(binding = 0) uniform UniformBufferObject 
//{
//    mat4 model;
//    mat4 view;
//    mat4 projection;
//} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
//layout(location = 2) in vec2 inUV;


layout(location = 0) out vec3 pixelColor;
//layout(location = 1) out vec2 pixelUV;

void main() 
{
    gl_PointSize = 14.0f;
    gl_Position = vec4(inPosition.xy, 1.0, 1.0);
    pixelColor = inColor.rgb;
	//pixelUV = inUV;
}