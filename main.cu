#define _USE_MATH_DEFINES
#include <cmath>
#include "cmplx.h"
#include "fft.h"
#include "four.h"
#include "fields.h"
#include "tint.h"
#include "shear.h"
#include "output.h"

int main
    ( void
){
    int istep = 0;
    cureal time = 0;

    otime = 1.0;
    nwrite = 100, write_fields = true;
    noise = true;
    nx = 256, ny = 512;
    nu = 1e-3, kappa = 1e-5, sigma = 1.0;
    g = 1.0, rho0_prime = 1.0, rho0 = 1.0, rho_eps1 = 1e-2, rho_eps2 = rho_eps1 / 2.0; 
    eps = 1e-10;
    delt = 1e-3, tmax = 30, nrst = 3, nst = 3;
    Lx = M_PI, Ly = M_PI;
    nthread = 1024;

    dx = Lx/nx, dy = Ly/ny;

    init_four();
    init_fields();
    init_tint();
    init_shear();
    init_output();

    initialize();

    if( write_fields ){
        file_out( istep, time );
        en_spectral( istep, time );
    }

    while( time <= tmax ){
        time_advance( istep, time );

        if( write_fields && time > next_otime ){
            next_otime += otime;
            file_out( istep, time );
            en_spectral( istep, time );
        }
    }


    finish_output();
    finish_shear();
    finish_tint();
    finish_fields();
    finish_four();

    return 0;
}
