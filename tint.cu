#include <cstdio>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <curand_kernel.h>
#include "cmplx.h"
#include "fft.h"
#include "fields.h"
#include "four.h"
#include "shear.h"

int nrst, nst;
bool noise;

namespace{
    cureal  *dv_rtmp, *ff;
    cucmplx *dv_ctmp;
}

/* ---------------------------------------------------------------- */

void init_tint( void );
void finish_tint( void );
__global__ static void disturb( cucmplx*, cureal* );
void init_dis( void );
void initialize( void );
__global__ static void eulerf_rho( cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab2_rho( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab2_rho( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab3_rho( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab3_rho( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, cureal );
__global__ static void add_phi( cucmplx*, const cucmplx* );
static void advect_rho( const int );
__global__ static void eulerb_rho( cucmplx*, const cureal );
__global__ static void bdf2_rho( cucmplx*, const cureal );
static void dissipate_rho( const int);
__global__ static void eulerf_omg( cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab2_omg( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab2_omg( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab3_omg( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void ab3_omg( cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cucmplx*, const cureal );
__global__ static void add_rho( cucmplx*, const cucmplx* );
static void advect_omg( const int );
__global__ static void eulerb_omg( cucmplx*, const cureal );
__global__ static void bdf2_omg( cucmplx*, const cureal );
static void dissipate_omg( const int );
static renew_fields( void );
static cureal maxval( cureal* );
static void check_cfl( cureal& );
void time_advance( int&, cureal& );

/* ---------------------------------------------------------------- */

void init_tint
    ( void
){
    cudaMalloc( (void**)&dv_rtmp, sizeof(cureal)*nx*ny );
    cudaMalloc( (void**)&dv_ctmp, sizeof(cucmplx)*nkx*nky );
    ff = new cureal [nx*ny];
}

void finish_tint
    ( void
){
    cudaFree( dv_rtmp );
    cudaFree( dv_ctmp );
    delete[] ff;
}

__global__ static void disturb
    ( cucmplx *dv_rho
    , cureal  *dv_dis
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        dv_rho[tid].x = dv_rho[tid].x + dv_dis[tid];
        dv_rho[tid].y = dv_rho[tid].y + dv_dis[tid];
    }
}

void init_dis
    ( void
){
    cureal *fk;
    cureal *dv_dis;
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky)+nthread-1/nthread );

    fk = new cureal [nkx*nky];
    cudaMalloc( (void**)&dv_dis, sizeof(cureal) * nkx *nky );

    srand( 10000 );
    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            if( abs(kx[ikx]) <= 40 && abs(ky[iky]) <= 80 ){
                fk[ikx*nky+iky] = rho_eps2 * (2 * ((double)rand() / (1.0 + RAND_MAX)) - 1.0) / (nx*ny);
            }
            else{
                fk[ikx*nky+iky] = 0;
            }
        }
    }

    cudaMemcpy( dv_dis, fk, sizeof(cureal) * nkx *nky, cudaMemcpyHostToDevice );
    disturb <<< cgrid, block >>> ( dv_arho0, dv_dis );

    delete[] fk;
    cudaFree( dv_dis );
}

void initialize
    ( void
){
    int ix ,iy;
    dim3 block( nthread );
    dim3 cgrid1( (nkx*nky)+nthread-1/nthread );
    dim3 cgrid2( ((nkx*nky-1)+nthread-1)/nthread );

    for( ix = 0; ix <= nx; ix++ ) xx[ix] = ix * dx;
    for( iy = 0; iy <= ny; iy++ ) yy[iy] = iy * dy;
    cudaMemcpy( dv_xx, xx, sizeof(cureal) * (nx+1), cudaMemcpyHostToDevice );
    cudaMemcpy( dv_yy, yy, sizeof(cureal) * (ny+1), cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_xx, &dv_xx, sizeof(dv_xx) );
    cudaMemcpyToSymbol( gb_yy, &dv_yy, sizeof(dv_yy) );

    for( ix = 0; ix < nx; ix++ ){
        for( iy = 0; iy < ny; iy++ ){
            omgz[ix*ny+iy] = 0;
            rho[ix*ny+iy] = rho_eps1*cos(2*xx[ix] + 10*yy[iy]);
            /* rho[ix*ny+iy] = sin(yy[iy]); */
        }
    }
    cudaMemcpy( dv_omgz, omgz, sizeof(cureal)*nx*ny, cudaMemcpyHostToDevice );
    cudaMemcpy( dv_rho, rho, sizeof(cureal)*nx*ny, cudaMemcpyHostToDevice );
    xtok( dv_omgz, dv_aomg0 );
    xtok( dv_rho, dv_arho0 );

    if( noise ) init_dis();

    neg_lapinv <<< cgrid2, block >>> ( dv_aomg0, dv_aphi );
    get_vector( dv_aphi, dv_vx, dv_vy );
}

__global__ static void eulerf_rho
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ) 
        fa0[tid] = fa1[tid]
                 + delt * dadt0[tid];
}

__global__ static void ab2_rho
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = fa1[tid]
                 + delt * (1.5*dadt0[tid] - 0.5*dadt1[tid]);
    }
}

__global__ static void ab2_rho
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *fa2
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = (4.0*fa1[tid] - fa2[tid]) / 3.0 
                 + (2.0/3.0)*delt * (2.0*dadt0[tid] - dadt1[tid]); 
    }
}

__global__ static void ab3_rho
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cucmplx *dadt2
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = fa1[tid] 
                 + (delt/12.0) * (23.0*dadt0[tid]-16.0*dadt1[tid]+5.0*dadt2[tid]);
    }
}

__global__ static void ab3_rho
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *fa2
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cucmplx *dadt2
    , const cureal   delt 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = (4.0*fa1[tid] - fa2[tid]) / 3.0 
                 + (delt/9.0) * (16.0*dadt0[tid]-14.0*dadt1[tid]+4.0*dadt2[tid]);
    }
}

__global__ static void add_phi
    (       cucmplx *dadt0
    , const cucmplx *aphi
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid / ct_nky;

    if( tid < ct_nkx*ct_nky )
        dadt0[tid] = dadt0[tid] + ct_rho0_prime * CIMAG * gb_kx[xid] * aphi[tid];
}

static void advect_rho
    ( const int istep
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    poisson_bracket_shear( dv_vx, dv_vy, dv_arho1, dv_rtmp );
    negative <<< rgrid, block >>> ( dv_rtmp );
    xtok( dv_rtmp, dv_drho0 );
    add_phi <<< cgrid, block >>> ( dv_drho0, dv_aphi );

    switch( nrst ){
        case 1:
            eulerf_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_drho0, delt );
            break;
        case 2:
            switch( istep ){
                case 0:
                    eulerf_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_drho0, 
                                                      delt );
                    break;
                default:
                    if( nu < eps ){
                        ab2_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, 
                                                       dv_drho0, dv_drho1, 
                                                       delt );
                    }
                    else{
                        ab2_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_arho2, 
                                                       dv_drho0, dv_drho1, 
                                                       delt );
                    }
                    break;
            }
        case 3:
            switch( istep ){
                case 0:
                    eulerf_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_drho0, 
                                                      delt );
                    break;
                case 1:
                    if( nu < eps ){
                        ab2_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, 
                                                       dv_drho0, dv_drho1, 
                                                       delt );
                    }
                    else{
                        ab2_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_arho2, 
                                                       dv_drho0, dv_drho1, 
                                                       delt );
                    }
                    break;
                default:
                    if( nu < eps ){
                        ab3_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, 
                                                       dv_drho0, dv_drho1, dv_drho2, 
                                                       delt );
                    }
                    else{
                        ab3_rho <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_arho2, 
                                                       dv_drho0, dv_drho1, dv_drho2, 
                                                       delt );
                    }
                    break;
            }
    }
}

__global__ static void eulerb_rho
    (       cucmplx *fa0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky )
        fa0[tid] = fa0[tid] / ( 1.0 + delt * ct_kappa * gb_kperp2_shear[tid] );
}

__global__ static void bdf2_rho
    (       cucmplx *fa0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky )
        fa0[tid] = fa0[tid] / ( 1.0 + (2.0/3.0) * delt * ct_kappa * gb_kperp2_shear[tid] );
}

static void dissipate_rho
    ( const int istep
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    switch( nst ){
        case 2:
            if( kappa < eps ) break;
            else{
                eulerb_rho <<< cgrid, block >>> ( dv_arho0, delt );
                break;
            }
        case 3:
            switch( istep ){
                case 0:
                    if( kappa < eps ) break;
                    else{
                        eulerb_rho <<< cgrid, block >>> ( dv_arho0, delt );
                        break;
                    }
                default:
                    if( kappa < eps ) break;
                    else{
                        bdf2_rho <<< cgrid, block >>> ( dv_arho0, delt );
                        break;
                    }
            }
    }
}

__global__ static void eulerf_omg
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ) 
        fa0[tid] = fa1[tid] 
                 + delt * dadt0[tid];
}

__global__ static void ab2_omg
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = fa1[tid] 
                 + delt * (1.5*dadt0[tid]-0.5*dadt1[tid]);
    }
}

__global__ static void ab2_omg
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *fa2
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = (4.0*fa1[tid] - fa2[tid]) / 3.0 
                 + (2.0/3.0)*delt * (2.0*dadt0[tid]-dadt1[tid]);
    }
}

__global__ static void ab3_omg
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cucmplx *dadt2
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = fa1[tid] 
                 + (delt/12.0) * (23.0*dadt0[tid]-16.0*dadt1[tid]+5.0*dadt2[tid]);
    }
}

__global__ static void ab3_omg
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *fa2
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cucmplx *dadt2
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        fa0[tid] = (4.0*fa1[tid] - fa2[tid]) / 3.0 
                 + (delt/9.0) * (16.0*dadt0[tid]-14.0*dadt1[tid]+4.0*dadt2[tid]);
    }
}

__global__ static void add_rho
    (       cucmplx *dadt0
    , const cucmplx *arho1
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid / ct_nky;

    if( tid < ct_nkx*ct_nky )
        dadt0[tid] = dadt0[tid] - ct_g / ct_rho0 * CIMAG * gb_kx[xid] * arho1[tid];
}

static void advect_omg
    ( const int istep
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    poisson_bracket_shear( dv_vx, dv_vy, dv_aomg1, dv_rtmp );
    negative <<< rgrid, block >>> ( dv_rtmp );
    xtok( dv_rtmp, dv_domg0 );
    add_rho <<< cgrid, block >>> ( dv_domg0, dv_arho1 );

    switch( nrst ){
        case 1:
            eulerf_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_domg0, 
                                              delt );
            break;
        case 2:
            switch( istep ){
                case 0:
                    eulerf_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_domg0, 
                                                      delt );
                    break;
                default:
                    if( nu < eps ){
                        ab2_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, 
                                                       dv_domg0, dv_domg1, 
                                                       delt );
                    }
                    else{
                        ab2_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_aomg2, 
                                                       dv_domg0, dv_domg1, 
                                                       delt );
                    }
                    break;
            }
        case 3:
            switch( istep ){
                case 0:
                    eulerf_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_domg0, 
                                                      delt );
                    break;
                case 1:
                    if( nu < eps ){
                        ab2_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, 
                                                       dv_domg0, dv_domg1, 
                                                       delt );
                    }
                    else{
                        ab2_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_aomg2, 
                                                       dv_domg0, dv_domg1, 
                                                       delt );
                    }
                    break;
                default:
                    if( nu < eps ){
                        ab3_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, 
                                                       dv_domg0, dv_domg1, dv_domg2, 
                                                       delt );
                    }
                    else{
                        ab3_omg <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_aomg2, 
                                                       dv_domg0, dv_domg1, dv_domg2, 
                                                       delt );
                    }
                    break;
            }
    }
}

__global__ static void eulerb_omg
    (       cucmplx *fa0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky )
        fa0[tid] = fa0[tid] / ( 1.0 + delt * ct_nu * gb_kperp2_shear[tid] );
}

__global__ static void bdf2_omg
    (       cucmplx *fa0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky )
        fa0[tid] = fa0[tid] / ( 1.0 + (2.0/3.0)*delt * ct_nu * gb_kperp2_shear[tid] );
}

static void dissipate_omg
    ( const int istep
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    switch( nst ){
        case 2:
            if( nu < eps ) break;
            else{
                eulerb_omg <<< cgrid, block >>> ( dv_aomg0, delt );
                break;
            }
        case 3:
            switch( istep ){
                case 0:
                    if( nu < eps ) break;
                    else{
                        eulerb_omg <<< cgrid, block >>> ( dv_aomg0, delt );
                        break;
                    }
                default:
                    if( nu < eps ) break;
                    else{
                        bdf2_omg <<< cgrid, block >>> ( dv_aomg0, delt );
                        break;
                }
            }
    }
}

static void renew_fields
    ( void
){
    cucmplx *tmp;

    tmp = dv_aomg2; dv_aomg2 = dv_aomg1; dv_aomg1 = dv_aomg0; dv_aomg0 = tmp;
    tmp = dv_domg2; dv_domg2 = dv_domg1; dv_domg1 = dv_domg0; dv_domg0 = tmp;

    tmp = dv_arho2; dv_arho2 = dv_arho1; dv_arho1 = dv_arho0; dv_arho0 = tmp;
    tmp = dv_drho2; dv_drho2 = dv_drho1; dv_drho1 = dv_drho0; dv_drho0 = tmp;
}

static cureal maxval
    ( cureal *ff
){
    cureal maxvalue;

    maxvalue = 0;
    for( int ix = 0; ix < nx; ix++ ){
        for( int iy = 0; iy < ny; iy++ )
            if( maxvalue < fabs(ff[ix*ny+iy]) )
                maxvalue = fabs(ff[ix*ny+iy]);
    }

    return maxvalue;
}

static void check_cfl
    ( cureal &delt
){
    dim3 block( nthread );
    dim3 cgrid( ((nkx*nky-1)+nthread-1)/nthread );

    neg_lapinv_shear <<< cgrid, block >>> ( dv_aomg0, dv_aphi );
    get_vector_shear( dv_aphi, dv_vx, dv_vy );

    cudaMemcpy( ff, dv_vx, sizeof(cureal)*nx*ny, cudaMemcpyDeviceToHost );
    cfl_vx = maxval( ff );
    while( (cfl_vx*delt/dx) > 0.1 ){
        delt /= 2.0;
        printf(": delt = %g\n", delt);
    }

    cudaMemcpy( ff, dv_vy, sizeof(cureal)*nx*ny, cudaMemcpyDeviceToHost );
    cfl_vy = maxval( ff );
    while( (cfl_vy*delt/dy) > 0.1 ){
        delt /= 2.0;
        printf(": delt = %g\n", delt);
    }
}

void time_advance
    ( int    &istep
    , cureal &time
){
    check_cfl( delt );
    renew_fields();

    advect_rho( istep );
    dissipate_rho( istep );

    advect_omg( istep );
    dissipate_omg( istep );

    update_shear( delt );

    istep++;
    time += delt;
}
