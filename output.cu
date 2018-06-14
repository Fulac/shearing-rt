#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include "cmplx.h"
#include "fft.h"
#include "four.h"
#include "fields.h"
#include "tint.h"
#include "shear.h"

bool   write_fields;
int    nwrite;
cureal otime, next_otime;

namespace{
    FILE *fp;
    char *filename, *filename_phi, *filename_ensp;
    char *filename_kspace;
    cucmplx *aomgz, *aphi, *arho;
    cureal  *ensp_ao, *ensp_ap, *ensp_ar;
}

/* ---------------------------------------------------------------------------------------------- */

void init_output( void );
void finish_output( void );
void file_out( int, cureal );
static cureal phi_max( cureal* );
void en_spectral( int, cureal );

/* ---------------------------------------------------------------------------------------------- */

void init_output
    ( void 
){
    next_otime = otime;
    filename = new char[15];
    filename_phi = new char[15];

    filename_kspace = new char[15];
    filename_ensp   = new char[15];
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
    delete[] filename_phi;
    delete[] filename_ensp;
    delete[] filename_kspace;
    delete[] aomgz;
    delete[] aphi;
    delete[] arho;
    delete[] ensp_ao;
    delete[] ensp_ap;
    delete[] ensp_ar;
}

void file_out
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


    sprintf( filename_phi, "phi_hat.dat" );
    if( (fp=fopen(filename_phi, "a+")) == NULL ) exit(1);
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

    sprintf( filename_ensp, "ky_ensp_t%09.6f.dat", time );
    if( (fp=fopen(filename_ensp, "w+")) == NULL ) exit(1);

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


    sprintf( filename_ensp, "kx_ensp_t%09.6f.dat", time );
    if( (fp=fopen(filename_ensp, "w+")) == NULL ) exit(1);

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


    sprintf( filename_kspace, "ks_t%09.6f.dat", time );
    if( (fp=fopen(filename_kspace, "w+")) == NULL ) exit(1);

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
