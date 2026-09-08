#include "demo.h"
#include "cinder/app/RendererGl.h"

namespace
{
    void prepareSettings(ci::app::App::Settings *settings)
    {
        settings->setHighDensityDisplayEnabled(true);
        //settings->disableFrameRate();
        settings->setFrameRate(144);
    }
}

using ci::app::RendererGl;
CINDER_APP(nbody::Demo, RendererGl(RendererGl::Options().msaa(16)), prepareSettings)
