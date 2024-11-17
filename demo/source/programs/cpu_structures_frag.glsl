"#version 410\n"
"out vec4 oColor;"
"layout (location=0) in float f_potential;"
"void main(void)"
"{"
"    float percent = f_potential;"
"    float a = percent;"
"    vec4 potential_color = vec4(1-(a*.5), a, 0, .1 + .15 * a);"
"    oColor = potential_color;"
"}"