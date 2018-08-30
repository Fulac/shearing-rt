// C headers
#include <cstdio>
#include <cstdlib>
#include <cmath>

// CUDA headers
#include <cuda.h>
#include <curand_kernel.h>

// Local headers
#include "cmplx.h"
#include "fft.h"
#include "fields.h"
#include "fourier.h"
#include "shear.h"

#define RANDSEED 2000

/* ---------------------------------------------------------------------------------------------- */
/*  Global Variables Definition                                                                   */
/* ---------------------------------------------------------------------------------------------- */

// プログラム全体で使用する変数を定義
bool noise_flag; // 初期値に擾乱を入れる (true) or 入れない (false)
cureal cfl_num;

// このファイル内でのみ使用するグローバル変数を定義
namespace{
    cureal eps;

    cureal  *dv_rtmp, *ff;
    cureal  *dv_v0, *dv_vmax;
    cucmplx *dv_ctmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Prototype                                                                            */
/* ---------------------------------------------------------------------------------------------- */

void init_tint
    ( void
);

void finish_tint
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void initialize
    ( void
);

static void init_dis
    ( void
);

__global__ static void disturb
    ( cucmplx *dv_rho
    , cureal  *dv_dis1
    , cureal  *dv_dis2
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void time_advance
    ( int    &istep
    , cureal &time
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void check_cfl
    ( cureal &delt
    , cureal  time
    , int     istep
);

__global__ static void add_v0
    ( cureal *dv_vx
    , cureal *dv_out
);

cureal maxvalue_search
    ( cureal *dv_field
);

__global__ static void pl_max_search
    ( cureal *dv_field
    , cureal *dv_output
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void renew_fields
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void advect_omg
    ( const int istep
);

__global__ static void add_ddxrho
    (       cucmplx *dadt0
    , const cucmplx *arho1
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void advect_rho
    ( const int istep
);

__global__ static void add_ddxphi
    (       cucmplx *dadt0
    , const cucmplx *aphi
);

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ static void eulerf
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cureal   delt
);

__global__ static void ab2
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cureal   delt
);

__global__ static void ab2
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *fa2
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cureal   delt
);

__global__ static void ab3
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cucmplx *dadt2
    , const cureal   delt
);

__global__ static void ab3
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *fa2
    , const cucmplx *dadt0
    , const cucmplx *dadt1
    , const cucmplx *dadt2
    , const cureal   delt
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void dissipate_omg
    ( const int istep
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void dissipate_rho
    ( const int istep
);

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ static void eulerb
    (       cucmplx *fa0
    , const cureal   diss
    , const cureal   delt
);

__global__ static void bdf2
    (       cucmplx *fa0
    , const cureal   diss
    , const cureal   delt
);

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Definition                                                                           */
/* ---------------------------------------------------------------------------------------------- */

void init_tint
    ( void
){
    eps = 1e-10;

    cudaMalloc( (void**)&dv_rtmp, sizeof(cureal)*nx*ny );
    cudaMalloc( (void**)&dv_ctmp, sizeof(cucmplx)*nkx*nky );
    cudaMalloc( (void**)&dv_vmax, sizeof(cureal)*nx*ny/2.0 );
    cudaMalloc( (void**)&dv_v0,   sizeof(cureal)*ny );
    ff = new cureal [nx*ny];
}

void finish_tint
    ( void
){
    cudaFree( dv_rtmp );
    cudaFree( dv_ctmp );
    cudaFree( dv_vmax );
    cudaFree( dv_v0 );
    delete[] ff;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void initialize
    ( void
){
    int ix ,iy;
    dim3 block( nthread );
    dim3 cgrid( ((nkx*nky-1)+nthread-1)/nthread );

    for( ix = 0; ix <= nx; ix++ ) xx[ix] = ix * dx;
    for( iy = 0; iy <= ny; iy++ ) yy[iy] = iy * dy;
    cudaMemcpy( dv_xx, xx, sizeof(cureal) * (nx+1), cudaMemcpyHostToDevice );
    cudaMemcpy( dv_yy, yy, sizeof(cureal) * (ny+1), cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_xx, &dv_xx, sizeof(dv_xx) );
    cudaMemcpyToSymbol( gb_yy, &dv_yy, sizeof(dv_yy) );

    for( ix = 0; ix < nx; ix++ ){
        for( iy = 0; iy < ny; iy++ ){
            omgz[ix*ny+iy] = 0;
            rho[ix*ny+iy]  = rho_eps1 * cos(2*xx[ix] + 10*yy[iy]);
        }
    }
    cudaMemcpy( dv_omgz, omgz, sizeof(cureal)*nx*ny, cudaMemcpyHostToDevice );
    cudaMemcpy( dv_rho,  rho,  sizeof(cureal)*nx*ny, cudaMemcpyHostToDevice );
    xtok( dv_omgz, dv_aomg0 );
    xtok( dv_rho,  dv_arho0 );

    if( noise_flag ) init_dis();

    neg_lapinv <<< cgrid, block >>> ( dv_aomg0, dv_aphi );
    get_vector( dv_aphi, dv_vx, dv_vy );
}

static void init_dis
    ( void
){
    cureal *fk1, *fk2;
    cureal *dv_dis1, *dv_dis2;
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    fk1 = new cureal [nkx*nky];
    fk2 = new cureal [nkx*nky];
    cudaMalloc( (void**)&dv_dis1, sizeof(cureal) * nkx *nky );
    cudaMalloc( (void**)&dv_dis2, sizeof(cureal) * nkx *nky );

    srand( RANDSEED );
    for( int ikx = 0; ikx < nkx; ikx++ ){
        for( int iky = 0; iky < nky; iky++ ){
            // 複素共役を保たない
            /* if( fabs(kx[ikx]) <= 2 && fabs(ky[iky]) <= 10 ){ */
            /*     fk1[ikx*nky+iky] = rho_eps2 * (2 * ((double)rand() / (1.0 + RAND_MAX)) - 1.0) */
            /*                      / sqrt(2 * 10); */
            /*     fk2[ikx*nky+iky] = rho_eps2 * (2 * ((double)rand() / (1.0 + RAND_MAX)) - 1.0) */
            /*                      / sqrt(2 * 10); */
            /* } */
            // 複素共役を保つ
            if( fabs(kx[ikx]) <= 2 && fabs(ky[iky]) != 0 && fabs(ky[iky]) <= 10 ){
                fk1[ikx*nky+iky] = rho_eps2 * (2 * ((double)rand() / (1.0 + RAND_MAX)) - 1.0)
                                 / sqrt(2 * 10);
                fk2[ikx*nky+iky] = rho_eps2 * (2 * ((double)rand() / (1.0 + RAND_MAX)) - 1.0)
                                 / sqrt(2 * 10);
            }
            else{
                fk1[ikx*nky+iky] = 0;
                fk2[ikx*nky+iky] = 0;
            }
        }
    }

    cudaMemcpy( dv_dis1, fk1, sizeof(cureal) * nkx *nky, cudaMemcpyHostToDevice );
    cudaMemcpy( dv_dis2, fk2, sizeof(cureal) * nkx *nky, cudaMemcpyHostToDevice );
    disturb <<< cgrid, block >>> ( dv_arho0, dv_dis1, dv_dis2 );

    delete[] fk1;
    delete[] fk2;
    cudaFree( dv_dis1 );
    cudaFree( dv_dis2 );
}

__global__ static void disturb
    ( cucmplx *dv_rho
    , cureal  *dv_dis1
    , cureal  *dv_dis2
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ){
        dv_rho[tid].x = dv_rho[tid].x + dv_dis1[tid];
        dv_rho[tid].y = dv_rho[tid].y + dv_dis2[tid];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void time_advance
    ( int    &istep
    , cureal &time
){
    check_cfl( delt, time, istep );
    renew_fields();

    advect_omg( istep );
    dissipate_omg( istep );

    advect_rho( istep );
    dissipate_rho( istep );

    update_shear( delt );

    istep++;
    time += delt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void check_cfl
    ( cureal &delt
    , cureal  time
    , int     istep
){
    dim3 threads( nthread );
    dim3 cblocks( (nkx*nky+nthread-1)/nthread );

    neg_lapinv_shear <<< cblocks, threads >>> ( dv_aomg0, dv_aphi );
    get_vector_shear( dv_aphi, dv_vx, dv_vy );

    dim3 rblocks( (nx*ny+nthread-1)/nthread );
    cureal cfl_vec;

    add_v0 <<< rblocks, threads >>> ( dv_vx, dv_rtmp );
    cfl_vec = maxvalue_search( dv_rtmp );
    while( (cfl_vec * delt / dx) > cfl_num ){
        delt /= 2.0;
        printf( "istep = %d, time = %g, cfl_vx = %g, delt = %g\n"
                , istep, time, cfl_vec, delt );
    }

    cfl_vec = maxvalue_search( dv_vy );
    while( (cfl_vec * delt / dy) > cfl_num ){
        delt /= 2.0;
        printf( "istep = %d, time = %g, cfl_vy = %g, delt = %g\n"
                , istep, time, cfl_vec, delt );
    }

    cfl_vec = rho0 / g;
    while( (cfl_vec * delt / dx) > cfl_num ){
        delt /= 2.0;
        printf( "istep = %d, time = %g, cfl_(rho0/g) = %g, delt = %g\n"
                , istep, time, cfl_vec, delt );
    }

    cfl_vec = rho0_prime;
    while( (cfl_vec * delt / dx) > cfl_num ){
        delt /= 2.0;
        printf( "istep = %d, time = %g, cfl_rho0_prime = %g, delt = %g\n"
                , istep, time, cfl_vec, delt );
    }
}

__global__ static void add_v0
    ( cureal *dv_vx
    , cureal *dv_out
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = idx%ct_ny;

    if( idx < ct_nx * ct_ny ) dv_out[idx] = dv_vx[idx] + ct_sigma * gb_yy[yid];
}

cureal maxvalue_search
    ( cureal *dv_field
){
    int threads = nthread;
    int blocks  = (nx*ny+(2*nthread)-1) / (2*nthread); 

    pl_max_search <<< blocks, threads, sizeof(cureal)*threads >>> ( dv_field, dv_vmax );
    cudaMemcpy( ff, dv_vmax, sizeof(cureal)*blocks, cudaMemcpyDeviceToHost );

    cureal maxvalue = 0;
    for( int i = 0; i < blocks; i++ ) if( maxvalue < ff[i] ) maxvalue = ff[i];

    return maxvalue;
}

__global__ static void pl_max_search
    ( cureal *dv_field
    , cureal *dv_output
){
    int tid = threadIdx.x;
    int  id = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    extern __shared__ cureal sh_data[];

    sh_data[tid] = fabs( dv_field[id           ] );
    cureal val   = fabs( dv_field[id+blockDim.x] );

    sh_data[tid] = ( sh_data[tid] > val ) ? sh_data[tid] : val;
    __syncthreads();

    if( blockDim.x >= 1024 && tid < 512 ){
        sh_data[tid] = ( sh_data[tid] > sh_data[tid+512] ) ? sh_data[tid] : sh_data[tid+512];
        __syncthreads();
    }
    if( blockDim.x >= 512 && tid < 256 ){
        sh_data[tid] = ( sh_data[tid] > sh_data[tid+256] ) ? sh_data[tid] : sh_data[tid+256];
        __syncthreads();
    }
    if( blockDim.x >= 256 && tid < 128 ){
        sh_data[tid] = ( sh_data[tid] > sh_data[tid+128] ) ? sh_data[tid] : sh_data[tid+128];
        __syncthreads();
    }
    if( blockDim.x >= 128 && tid < 64){
        sh_data[tid] = ( sh_data[tid] > sh_data[tid+64] ) ? sh_data[tid] : sh_data[tid+64];
        __syncthreads();
    }

    if( tid < 32 ){
        if( blockDim.x >= 64 ){
            sh_data[tid] = ( sh_data[tid] > sh_data[tid+32] ) ? sh_data[tid] : sh_data[tid+32];
            __syncthreads();
        }
        if( blockDim.x >= 32 ){
            sh_data[tid] = ( sh_data[tid] > sh_data[tid+16] ) ? sh_data[tid] : sh_data[tid+16];
            __syncthreads();
        }
        if( blockDim.x >= 16 ){
            sh_data[tid] = ( sh_data[tid] > sh_data[tid+ 8] ) ? sh_data[tid] : sh_data[tid+ 8];
            __syncthreads();
        }
        if( blockDim.x >=  8 ){
            sh_data[tid] = ( sh_data[tid] > sh_data[tid+ 4] ) ? sh_data[tid] : sh_data[tid+ 4];
            __syncthreads();
        }
        if( blockDim.x >=  4 ){
            sh_data[tid] = ( sh_data[tid] > sh_data[tid+ 2] ) ? sh_data[tid] : sh_data[tid+ 2];
            __syncthreads();
        }
        if( blockDim.x >=  2 ){
            sh_data[tid] = ( sh_data[tid] > sh_data[tid+ 1] ) ? sh_data[tid] : sh_data[tid+ 1];
            __syncthreads();
        }
    }

    if( tid == 0 ) dv_output[blockIdx.x] = sh_data[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void renew_fields
    ( void
){
    cucmplx *tmp;

    tmp = dv_aomg2; dv_aomg2 = dv_aomg1; dv_aomg1 = dv_aomg0; dv_aomg0 = tmp;
    tmp = dv_domg2; dv_domg2 = dv_domg1; dv_domg1 = dv_domg0; dv_domg0 = tmp;

    tmp = dv_arho2; dv_arho2 = dv_arho1; dv_arho1 = dv_arho0; dv_arho0 = tmp;
    tmp = dv_drho2; dv_drho2 = dv_drho1; dv_drho1 = dv_drho0; dv_drho0 = tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void advect_omg
    ( const int istep
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    poisson_bracket_shear( dv_vx, dv_vy, dv_aomg1, dv_rtmp );
    negative <<< rgrid, block >>> ( dv_rtmp );
    xtok( dv_rtmp, dv_domg0 );
    add_ddxrho <<< cgrid, block >>> ( dv_domg0, dv_arho1 );

    switch( istep ){
        case 0:
            eulerf <<< cgrid, block >>> ( dv_aomg0, dv_aomg1
                                        , dv_domg0
                                        , delt );
            break;
        case 1:
            if( nu < eps ){
                ab2 <<< cgrid, block >>> ( dv_aomg0, dv_aomg1
                                         , dv_domg0, dv_domg1
                                         , delt );
            }
            else{
                ab2 <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_aomg2
                                         , dv_domg0, dv_domg1
                                         , delt );
            }
            break;
        default:
            if( nu < eps ){
                ab3 <<< cgrid, block >>> ( dv_aomg0, dv_aomg1
                                         , dv_domg0, dv_domg1, dv_domg2
                                         , delt );
            }
            else{
                ab3 <<< cgrid, block >>> ( dv_aomg0, dv_aomg1, dv_aomg2
                                         , dv_domg0, dv_domg1, dv_domg2
                                         , delt );
            }
            break;
    }
}

__global__ static void add_ddxrho
    (       cucmplx *dadt0
    , const cucmplx *arho1
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid / ct_nky;

    if( tid < ct_nkx*ct_nky )
        dadt0[tid] = dadt0[tid] - ct_g / ct_rho0 * CIMAG * gb_kx[xid] * arho1[tid];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void advect_rho
    ( const int istep
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    poisson_bracket_shear( dv_vx, dv_vy, dv_arho1, dv_rtmp );
    negative <<< rgrid, block >>> ( dv_rtmp );
    xtok( dv_rtmp, dv_drho0 );
    add_ddxphi <<< cgrid, block >>> ( dv_drho0, dv_aphi );

    switch( istep ){
        case 0:
            eulerf <<< cgrid, block >>> ( dv_arho0, dv_arho1
                                        , dv_drho0
                                        , delt );
            break;
        case 1:
            if( kappa < eps ){
                ab2 <<< cgrid, block >>> ( dv_arho0, dv_arho1
                                         , dv_drho0, dv_drho1
                                         , delt );
            }
            else{
                ab2 <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_arho2
                                         , dv_drho0, dv_drho1
                                         , delt );
            }
            break;
        default:
            if( kappa < eps ){
                ab3 <<< cgrid, block >>> ( dv_arho0, dv_arho1
                                         , dv_drho0, dv_drho1, dv_drho2
                                         , delt );
            }
            else{
                ab3 <<< cgrid, block >>> ( dv_arho0, dv_arho1, dv_arho2
                                         , dv_drho0, dv_drho1, dv_drho2
                                         , delt );
            }
            break;
    }
}

__global__ static void add_ddxphi
    (       cucmplx *dadt0
    , const cucmplx *aphi
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid / ct_nky;

    if( tid < ct_nkx*ct_nky )
        dadt0[tid] = dadt0[tid] + ct_rho0_prime * CIMAG * gb_kx[xid] * aphi[tid];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ static void eulerf
    (       cucmplx *fa0
    , const cucmplx *fa1
    , const cucmplx *dadt0
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ) 
        fa0[tid] = fa1[tid] + delt * dadt0[tid];
}

__global__ static void ab2
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

__global__ static void ab2
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

__global__ static void ab3
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

__global__ static void ab3
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

////////////////////////////////////////////////////////////////////////////////////////////////////

static void dissipate_omg
    ( const int istep
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    switch( istep ){
        case 0:
            if( nu < eps ) break;
            else{
                eulerb <<< cgrid, block >>> ( dv_aomg0, nu, delt );
                break;
            }
        default:
            if( nu < eps ) break;
            else{
                bdf2 <<< cgrid, block >>> ( dv_aomg0, nu, delt );
                break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void dissipate_rho
    ( const int istep
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    switch( istep ){
        case 0:
            if( kappa < eps ) break;
            else{
                eulerb <<< cgrid, block >>> ( dv_arho0, kappa, delt );
                break;
            }
        default:
            if( kappa < eps ) break;
            else{
                bdf2 <<< cgrid, block >>> ( dv_arho0, kappa, delt );
                break;
            }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ static void eulerb
    (       cucmplx *fa0
    , const cureal   diss
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky )
        fa0[tid] = fa0[tid] / ( 1.0 + delt * diss * gb_kperp2_shear[tid] );
}

__global__ static void bdf2
    (       cucmplx *fa0
    ,       cureal   diss
    , const cureal   delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky )
        fa0[tid] = fa0[tid] / ( 1.0 + (2.0/3.0)*delt * diss * gb_kperp2_shear[tid] );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
