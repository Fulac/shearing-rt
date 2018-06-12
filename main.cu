#define _USE_MATH_DEFINES

// C headers
#include <cmath>

// C++ headers
#include <string>

// Boost headers
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

// Local headers
#include "cmplx.h"
#include "fft.h"
#include "four.h"
#include "fields.h"
#include "tint.h"
#include "shear.h"
#include "output.h"

template <class T>
T readEntry
   ( boost::property_tree::ptree pt
   , std::string section
   , std::string name
   , T defaultValue );

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

    boost::property_tree::ptree pt;
    std::string tempstr;

    try{
        boost::property_tree::read_ini( "config.ini", pt );
    }
    catch( std::exception &e ){
        printf( "ERROR: unable to read config file: %s", e.what() );
        exit(1);
    }

    otime = readEntry<cureal>( pt, "output", "output time step", 1.0 );
    // TODO

    tempstr = readEntry<std::string>( pt, "simulation", "advection method", "ab3" );
    if( tempstr == "forward euler" ){
        nrst = 1;
    } else if( tempstr == "ab2" ){
        nrst = 2;
    } else if( tempstr == "ab3" ){
        nrst = 3;
    } else {
        printf( "ERROR: unknown advection method: %s\n", tempstr.c_str() );
        printf( "       Available methods: forward euler, ab2, ab3\n" );
        exit(1);
    }
    // TODO: dissipation method -> nst

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

template <class T>
T readEntry
   ( boost::property_tree::ptree pt
   , std::string section
   , std::string name
   , T           defaultValue
){
   T value;

   try {
      // get value
      value = pt.get<T>( section+"."+name );
   } catch( boost::property_tree::ptree_error &err ) {
      // show warning if key is missing
      printf( "WARNING: readEntry: Key \"%s\" in section [%s] not found. Using default.\n", name.c_str(), section.c_str() );
      value = defaultValue;
   }

   return value;
}

