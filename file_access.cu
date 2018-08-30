// C headers
#include <cstdio>
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

#define FILENAMELEN 256

/* ---------------------------------------------------------------------------------------------- */
/*  Global Variables Definition                                                                   */
/* ---------------------------------------------------------------------------------------------- */

// プログラム全体で使用する変数を定義
bool   write_fields;
int    nwrite;
cureal output_time, next_output_time;

// このファイル内でのみ使用するグローバル変数を定義
namespace{
    FILE *fp;
    char *filename;
    cucmplx *aomgz, *aphi, *arho;
    cureal  *ensp_ao, *ensp_ap, *ensp_ar;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Prototype                                                                            */
/* ---------------------------------------------------------------------------------------------- */

void input_data
    ( void
);

template <class T>
T readEntry
   ( boost::property_tree::ptree pt
   , std::string section
   , std::string name
   , T defaultValue
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_output
    ( void
);

void finish_output
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void output_fields
    ( const int    istep
    , const cureal time
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void output_maxamp
    ( const cureal time
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void en_spectral
    ( const int    istep
    , const cureal time
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void k_data
    ( const int flag
);

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Definition                                                                           */
/* ---------------------------------------------------------------------------------------------- */

void input_data
    ( void
){
    boost::property_tree::ptree pt;

    try{
        boost::property_tree::read_ini( "config.ini", pt );
    }
    catch( std::exception &e ){
        printf( "ERROR: unable to read config file: %s", e.what() );
        exit(1);
    }

    // simulation parameters
    nx      = readEntry<int>( pt, "simulation", "nx",  512 );
    ny      = readEntry<int>( pt, "simulation", "ny", 1024 );
    nthread = readEntry<int>( pt, "simulation", "cuda thread num", 1024 );
    Lx      = readEntry<cureal>( pt, "simulation", "Lx", M_PI );
    Ly      = readEntry<cureal>( pt, "simulation", "Ly", M_PI );
    delt    = readEntry<cureal>( pt, "simulation", "time step", 1e-3 );
    tmax    = readEntry<cureal>( pt, "simulation", "time max", 30 );
    cfl_num = readEntry<cureal>( pt, "simulation", "cfl number", 5e-3 );

    // output parameters
    output_time  = readEntry<cureal>( pt, "output", "output time step",  1.0 );
    nwrite       = readEntry<int>(    pt, "output", "output loop count", 100 );
    write_fields = readEntry<bool>(   pt, "output", "write output",      true );

    // problem parameters
    noise_flag = readEntry<bool>(   pt, "problem", "initial noise", true );
    nu         = readEntry<cureal>( pt, "problem", "nu",            1e-3 );
    kappa      = readEntry<cureal>( pt, "problem", "kappa",         1e-5 );
    sigma      = readEntry<cureal>( pt, "problem", "sigma",         1.0 );
    g          = readEntry<cureal>( pt, "problem", "g",             1.0 );
    rho0_prime = readEntry<cureal>( pt, "problem", "rho0_prime",    1.0 );
    rho0       = readEntry<cureal>( pt, "problem", "rho0",          1.0 );
    rho_eps1   = readEntry<cureal>( pt, "problem", "rho_eps1",      1e-2 );
    rho_eps2   = rho_eps1 / 2.0;
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
   }
   catch( boost::property_tree::ptree_error &err ) {
      // show warning if key is missing
      printf( "WARNING: readEntry: Key \"%s\" in section [%s] not found. Using default.\n"
            , name.c_str(), section.c_str() );
      value = defaultValue;
   }

   return value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_output
    ( void 
){
    next_output_time = output_time;
    filename = new char[FILENAMELEN];

    aomgz = new cucmplx[nkx*nky];
    aphi  = new cucmplx[nkx*nky];
    arho  = new cucmplx[nkx*nky];
    ensp_ao = new cureal[nky];
    ensp_ap = new cureal[nky];
    ensp_ar = new cureal[nky];
}

void finish_output
    ( void
){
    delete[] filename;
    delete[] aomgz;
    delete[] aphi;
    delete[] arho;
    delete[] ensp_ao;
    delete[] ensp_ap;
    delete[] ensp_ar;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void output_fields
    ( const int    istep
    , const cureal time 
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    printf("time = %g\n", time);

    if( sigma ){
        ktox_shear( dv_aomg0, dv_omgz );
        ktox_shear( dv_aphi,  dv_phi );
        ktox_shear( dv_arho0, dv_rho );
    }
    else{
        ktox( dv_aomg0, dv_omgz );
        ktox( dv_aphi,  dv_phi );
        ktox( dv_arho0, dv_rho );
    }

    cudaMemcpy( omgz, dv_omgz, sizeof(cureal)*nx*ny, cudaMemcpyDeviceToHost );
    cudaMemcpy( phi,  dv_phi,  sizeof(cureal)*nx*ny, cudaMemcpyDeviceToHost );
    cudaMemcpy( rho,  dv_rho,  sizeof(cureal)*nx*ny, cudaMemcpyDeviceToHost );

    snprintf( filename, FILENAMELEN, "n%05d_t%09.6f.dat", istep/nwrite, time );
    if( (fp=fopen(filename, "w+")) == NULL ) exit(1);
    for( int ix = 0; ix <= nx; ix++ ){
        for( int iy = 0; iy <= ny; iy++ ){
            if( ix < nx && iy < ny ){
                fprintf( fp, "%+e %+e %+e %+e %+e\n"
                       , xx[ix], yy[iy], omgz[ix*ny+iy], phi[ix*ny+iy], rho[ix*ny+iy] );
            }
            else if( ix == nx && iy == ny ){
                fprintf( fp, "%+e %+e %+e %+e %+e\n"
                       , xx[ix], yy[iy], omgz[0], phi[0], rho[0] );
            }
            else if( ix == nx ){
                fprintf( fp, "%+e %+e %+e %+e %+e\n"
                       , xx[ix], yy[iy], omgz[iy], phi[iy], rho[iy] );
            }
            else if( iy == ny ){
                fprintf( fp, "%+e %+e %+e %+e %+e\n"
                       , xx[ix], yy[iy], omgz[ix*ny], phi[ix*ny], rho[ix*ny] );
            }
        }
        fprintf( fp, "\n" );
    }
    fclose( fp );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void output_maxamp
    ( const cureal time
){
    if( sigma ){
        ktox_shear( dv_aphi,  dv_phi );
        ktox_shear( dv_arho0, dv_rho );
    }
    else{
        ktox( dv_aphi,  dv_phi );
        ktox( dv_arho0, dv_rho );
    }

    snprintf( filename, FILENAMELEN, "phi_hat.dat" );
    if( (fp=fopen(filename, "a+")) == NULL ) exit(1);
    fprintf( fp, "%.10f %+e\n", time, maxvalue_search(dv_phi) );
    fclose( fp );

    snprintf( filename, FILENAMELEN, "rho_hat.dat" );
    if( (fp=fopen(filename, "a+")) == NULL ) exit(1);
    fprintf( fp, "%.10f %+e\n", time, maxvalue_search(dv_rho) );
    fclose( fp );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void en_spectral
    ( const int    /*istep*/
    , const cureal time
){
    cureal re, im;
    cureal ao, ap, ar;

    cudaMemcpy( aomgz, dv_aomg0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );
    cudaMemcpy( aphi,  dv_aphi,  sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );
    cudaMemcpy( arho,  dv_arho0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );

    snprintf( filename, FILENAMELEN, "ky_ensp_t%09.6f.dat", time );
    if( (fp=fopen(filename, "w+")) == NULL ) exit(1);

    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            if( ikx == 0 ){
                re = aomgz[ikx*nky+iky].x;
                im = aomgz[ikx*nky+iky].y;
                ensp_ao[iky] = re*re + im*im;

                re = aphi[ikx*nky+iky].x;
                im = aphi[ikx*nky+iky].y;
                ensp_ap[iky] = re*re + im*im; 

                re = arho[ikx*nky+iky].x;
                im = arho[ikx*nky+iky].y;
                ensp_ar[iky] = re*re + im*im; 
            }
            else{
                re  = aomgz[ikx*nky+iky].x;
                im  = aomgz[ikx*nky+iky].y;
                ensp_ao[iky] += re*re + im*im;

                re  = aphi[ikx*nky+iky].x;
                im  = aphi[ikx*nky+iky].y;
                ensp_ap[iky] += re*re + im*im; 

                re  = arho[ikx*nky+iky].x;
                im  = arho[ikx*nky+iky].y;
                ensp_ar[iky] += re*re + im*im; 
            }
        }
    }
    for( int iky = 0; iky < nky; iky++ ) 
        fprintf( fp, "%+e %+e %+e %+e\n", 
                 ky[iky], ensp_ao[iky]/nkx, ensp_ap[iky]/nkx, ensp_ar[iky]/nkx );
    fclose( fp );


    snprintf( filename, FILENAMELEN, "kx_ensp_t%09.6f.dat", time );
    if( (fp=fopen(filename, "w+")) == NULL ) exit(1);

    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            if( iky == 0 ){
                re = aomgz[ikx*nky+iky].x;
                im = aomgz[ikx*nky+iky].y;
                ensp_ao[ikx] = re*re + im*im;

                re = aphi[ikx*nky+iky].x;
                im = aphi[ikx*nky+iky].y;
                ensp_ap[ikx] = re*re + im*im; 

                re = arho[ikx*nky+iky].x;
                im = arho[ikx*nky+iky].y;
                ensp_ar[ikx] = re*re + im*im; 
            }
            else{
                re  = aomgz[ikx*nky+iky].x;
                im  = aomgz[ikx*nky+iky].y;
                ensp_ao[ikx] += re*re + im*im;

                re  = aphi[ikx*nky+iky].x;
                im  = aphi[ikx*nky+iky].y;
                ensp_ap[ikx] += re*re + im*im; 

                re  = arho[ikx*nky+iky].x;
                im  = arho[ikx*nky+iky].y;
                ensp_ar[ikx] += re*re + im*im; 
            }
        }
    }
    for( int ikx = 0; ikx < nkx; ikx++ ) 
        fprintf( fp, "%+e %+e %+e %+e\n", 
                 ky[ikx], ensp_ao[ikx]/nky, ensp_ap[ikx]/nky, ensp_ar[ikx]/nky );
    fclose( fp );


    snprintf( filename, FILENAMELEN, "ks_t%09.6f.dat", time );
    if( (fp=fopen(filename, "w+")) == NULL ) exit(1);

    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            re = aomgz[ikx*nky+iky].x;
            im = aomgz[ikx*nky+iky].y;
            ao = sqrt( re*re + im*im ); 

            re = aphi[ikx*nky+iky].x;
            im = aphi[ikx*nky+iky].y;
            ap = sqrt( re*re + im*im ); 

            re = arho[ikx*nky+iky].x;
            im = arho[ikx*nky+iky].y;
            ar = sqrt( re*re + im*im ); 

            fprintf( fp, "%+e %+e %+e %+e %+e\n",
                     kx[ikx], ky[iky], ao, ap, ar );
        }
        fprintf( fp, "\n" );
    }
    fclose( fp );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void k_data
    ( const int flag
){
    cudaMemcpy( aomgz, dv_aomg0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );
    cudaMemcpy( aphi,  dv_aphi,  sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );
    cudaMemcpy( arho,  dv_arho0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );

    snprintf( filename, FILENAMELEN, "k_data_f%d.dat", flag );
    if( (fp=fopen(filename, "w+")) == NULL ) exit(1);

    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            fprintf( fp, "%+e ", arho[ikx*nky+iky].x );
        }
        fprintf( fp, "\n" );
    }
    fprintf( fp, "\n" );
    fprintf( fp, "\n" );
    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            fprintf( fp, "%+e ", arho[ikx*nky+iky].y );
        }
        fprintf( fp, "\n" );
    }
    fclose( fp );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
