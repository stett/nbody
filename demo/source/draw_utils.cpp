#include "draw_utils.h"
#include "cinder/gl/gl.h"

using namespace ci;

namespace util {
    void draw_axes(const float size) {
        gl::color(1, .2, .2, .5);
        gl::drawLine(vec3(-size, 0, 0), vec3(size, 0, 0));
        gl::color(.2, 1, .2, .5);
        gl::drawLine(vec3(0, -size, 0), vec3(0, size, 0));
        gl::color(.2, .2, 1, .5);
        gl::drawLine(vec3(0, 0, -size), vec3(0, 0, size));
    }
}
