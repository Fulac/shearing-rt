// Local headers
#include "cmplx.h"
#include "fft.h"
#include "fourier.h"
#include "fields.h"
#include "time_integral.h"
#include "shear.h"
#include "file_access.h"

int main
    ( void
){
    int istep = 0;
    cureal time = 0;

    cureal output_maxamp_time = 0.1;
    cureal next_output_maxamp_time = output_maxamp_time;

    input_data();

    dx = Lx/nx, dy = Ly/ny;

    init_four();
    init_fields();
    init_tint();
    init_shear();
    init_output();

    initialize();

    if( write_fields ){
        output_fields( istep, time );
        output_maxamp( time );
        en_spectral( istep, time );
    }

    while( time <= tmax ){
        time_advance( istep, time );

        if( write_fields && time > next_output_time ){
            next_output_time += output_time;
            output_fields( istep, time );
            en_spectral( istep, time );
        }
        if( write_fields && time > next_output_maxamp_time ){
            next_output_maxamp_time += output_maxamp_time;
            output_maxamp( time );
        }
    }

    finish_output();
    finish_shear();
    finish_tint();
    finish_fields();
    finish_four();

    return 0;
}
