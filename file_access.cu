// C headers
#include <cstdio>
#include <cmath>

// C++ headers
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

// Local headers
#include "cmplx.h"
#include "fft.h"
#include "fourier.h"
#include "fields.h"
#include "time_integral.h"
#include "shear.h"

bool   write_fields;
int    nwrite;
cureal otime, next_otime;

namespace{
    FILE *fp;
    char *filename;
    cucmplx *aomgz, *aphi, *arho;
    cureal  *ensp_ao, *ensp_ap, *ensp_ar;
}

/* ---------------------------------------------------------------------------------------------- */

void input_data( void );
template <class T>
T readEntry
   ( boost::property_tree::ptree pt
   , std::string section
   , std::string name
   , T defaultValue );
void init_output( void );
void finish_output( void );
void output_fields( int, cureal );
static cureal phi_max( cureal* );
void en_spectral( int, cureal );

/* ---------------------------------------------------------------------------------------------- */

void input_data
    ( void
){
    boost::property_tree::ptree pt;
    std::string tempstr;

    try{
        boost::property_tree::read_ini( "config.ini", pt );
    }
    catch( std::exception &e ){
        printf( "ERROR: unable to read config file: %s", e.what() );
        exit(1);
    }

    // simulation parameters
    nx = readEntry<int>( pt, "simulation", "nx",  512 );
    ny = readEntry<int>( pt, "simulation", "ny", 1024 );
    Lx = readEntry<cureal>( pt, "simulation", "Lx", M_PI );
    Ly = readEntry<cureal>( pt, "simulation", "Ly", M_PI );
    eps = readEntry<cureal>( pt, "simulation", "eps", 1e-10 );
    delt = readEntry<cureal>( pt, "simulation", "time step", 1e-3 );
    tmax = readEntry<cureal>( pt, "simulation", "time max", 30 );
    nthread = readEntry<cureal>( pt, "simulation", "cuda thread num", 1024 );

    tempstr = readEntry<std::string>( pt, "simulation", "advection method", "ab3" );
    if( tempstr == "forward euler" ){
        nrst = 1;
    }
    else if( tempstr == "ab2" ){
        nrst = 2;
    }
    else if( tempstr == "ab3" ){
        nrst = 3;
    }
    else{
        printf( "ERROR: unknown advection method: %s\n", tempstr.c_str() );
        printf( "       Available methods: forward euler, ab2, ab3\n" );
        exit(1);
    }

    tempstr = readEntry<std::string>( pt, "simulation", "dissipation method", "bdf2" );
    if( tempstr == "backward euler" ){
        nst = 2;
    }
    else if( tempstr == "bdf2" ){
        nst = 3;
    }
    else{
        printf( "ERROR: unknown dissipation method: %s\n", tempstr.c_str() );
        printf( "       Available methods: backward euler, bdf2\n" );
        exit(1);
    }

    // output parameters
    otime = readEntry<cureal>( pt, "output", "output time step", 1.0 );
    nwrite = readEntry<int>( pt, "output", "output loop count", 100 );
    write_fields = readEntry<bool>( pt, "output", "write output", true );

    // problem parameters
    noise = readEntry<bool>( pt, "problem", "initial noise", true );
    nu = readEntry<cureal>( pt, "problem", "nu", 1e-3 );
    kappa = readEntry<cureal>( pt, "problem", "kappa", 1e-5 );
    sigma = readEntry<cureal>( pt, "problem", "sigma", 1.0 );
    g = readEntry<cureal>( pt, "problem", "g", 1.0 );
    rho0_prime = readEntry<cureal>( pt, "problem", "rho0_prime", 1.0 );
    rho0 = readEntry<cureal>( pt, "problem", "rho0", 1.0 );
    rho_eps1 = readEntry<cureal>( pt, "problem", "rho_eps1", 1e-2 );
    rho_eps2 = readEntry<cureal>( pt, "problem", "rho_eps2", rho_eps1/2.0 );
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

void init_output
    ( void 
){
    next_otime = otime;
    filename = new char[15];

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

void output_fields
    ( int    istep
    , cureal time 
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

    sprintf( filename, "n%05d_t%09.6f.dat", istep/nwrite, time );
    if( (fp=fopen(filename, "w+")) == NULL ) exit(1);
    for( int ix = 0; ix < nx; ix++ ){
        for( int iy = 0; iy < ny; iy++ ){
            fprintf( fp, "%g %g %g %g %g\n"
                   , xx[ix], yy[iy], omgz[ix*ny+iy], phi[ix*ny+iy], rho[ix*ny+iy] );
        }
        fprintf( fp, "\n" );
    }
    fclose( fp );


    sprintf( filename, "phi_hat.dat" );
    if( (fp=fopen(filename, "a+")) == NULL ) exit(1);
    fprintf( fp, "%.10f %g\n", time, phi_max(phi) );
    fclose( fp );
}

static cureal phi_max
    ( cureal *phi
){
    cureal max_val = 0;

    for( int ix = 0; ix < nx; ix++ ){
        for( int iy = 0; iy < ny; iy++ ){
            if( max_val < fabs(phi[ix*ny+iy]) ) max_val = fabs(phi[ix*ny+iy]);
        }
    }

    return max_val;
}

void en_spectral
    ( int istep
    , cureal time
){
    cureal re, im;
    cureal ao, ap, ar;

    cudaMemcpy( aomgz, dv_aomg0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );
    cudaMemcpy( aphi,  dv_aphi,  sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );
    cudaMemcpy( arho,  dv_arho0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToHost );

    sprintf( filename, "ky_ensp_t%09.6f.dat", time );
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
        fprintf( fp, "%g %g %g %g\n", 
                 ky[iky], ensp_ao[iky]/nkx, ensp_ap[iky]/nkx, ensp_ar[iky]/nkx );
    fclose( fp );


    sprintf( filename, "kx_ensp_t%09.6f.dat", time );
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
        fprintf( fp, "%g %g %g %g\n", 
                 ky[iky], ensp_ao[iky]/nkx, ensp_ap[iky]/nkx, ensp_ar[iky]/nkx );
    fclose( fp );


    sprintf( filename, "ks_t%09.6f.dat", time );
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

            fprintf( fp, "%g %g %g %g %g\n",
                     kx[ikx], ky[iky], ao, ap, ar );
        }
        fprintf( fp, "\n" );
    }
    fclose( fp );
}
