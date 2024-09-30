#version 330
#pragma vscode_glsllint_stage : vert

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;   // rgb
layout(location = 2) in vec3 normal;  // normal

out vec3 world_frag_pos;
out vec3 cam_frag_pos;
out vec3 vert_color;   // pass through
out vec3 vert_normal;  // pass through

void main() {
    vec4 pos = vec4(position, 1.0);  // in object space
    vec4 norm = vec4(normal, 1.0);   // in object space
    vec4 world = M * pos;            // in world space
    vec4 cam = V * world;            // in camera space

    // Outputs
    world_frag_pos = vec3(world);
    cam_frag_pos = vec3(cam);
    gl_Position = K * cam;  // doing a perspective projection to clip space

    vert_color = color;                                    // passing through
    vert_normal = vec3(transpose(inverse(V * M)) * norm);  // in camera space
}