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
#include "fourier.h"
#include "fields.h"
#include "time_integral.h"
#include "shear.h"
#include "file_access.h"

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
        en_spectral( istep, time );
    }

    while( time <= tmax ){
        time_advance( istep, time );

        if( write_fields && time > next_otime ){
            next_otime += otime;
            output_fields( istep, time );
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
