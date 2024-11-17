"#version 410\n"
"uniform mat4	ciModelViewProjection;"
"layout (points) in;"
"layout (line_strip, max_vertices = 18) out;"
"layout (location=0) in vec3 g_min_pos[];"
"layout (location=1) in vec3 g_max_pos[];"
"layout (location=2) in float g_potential[];"
"layout (location=0) out float f_potential;"
"void main(void)"
"{"
"    f_potential = g_potential[0];"
"    vec3 aa = g_min_pos[0];"
"    vec3 bb = g_max_pos[0];"
"    vec4 v[8] = vec4[8]("
"       ciModelViewProjection * vec4(aa.x, aa.y, aa.z, 1),"
"       ciModelViewProjection * vec4(bb.x, aa.y, aa.z, 1),"
"       ciModelViewProjection * vec4(bb.x, bb.y, aa.z, 1),"
"       ciModelViewProjection * vec4(aa.x, bb.y, aa.z, 1),"
"       ciModelViewProjection * vec4(aa.x, aa.y, bb.z, 1),"
"       ciModelViewProjection * vec4(bb.x, aa.y, bb.z, 1),"
"       ciModelViewProjection * vec4(bb.x, bb.y, bb.z, 1),"
"       ciModelViewProjection * vec4(aa.x, bb.y, bb.z, 1));"
""
"    gl_Position = v[0]; EmitVertex();"
"    gl_Position = v[0+4]; EmitVertex();"
"    gl_Position = v[0]; EmitVertex();"
"    gl_Position = v[1]; EmitVertex();"
"    gl_Position = v[1+4]; EmitVertex();"
"    gl_Position = v[1]; EmitVertex();"
"    gl_Position = v[2]; EmitVertex();"
"    gl_Position = v[2+4]; EmitVertex();"
"    gl_Position = v[2]; EmitVertex();"
"    gl_Position = v[3]; EmitVertex();"
"    gl_Position = v[3+4]; EmitVertex();"
"    gl_Position = v[3]; EmitVertex();"
"    gl_Position = v[0]; EmitVertex();"
"    gl_Position = v[4]; EmitVertex();"
"    gl_Position = v[5]; EmitVertex();"
"    gl_Position = v[6]; EmitVertex();"
"    gl_Position = v[7]; EmitVertex();"
"    gl_Position = v[4]; EmitVertex();"
"    EndPrimitive();"
"}"