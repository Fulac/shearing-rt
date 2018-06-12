#include <cuda.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include "cmplx.h"
#include "fft.h"

cureal *kx, *ky, *kperp2, *dv_kx, *dv_ky, *dv_kperp2;
__device__ cureal *gb_kx, *gb_ky, *gb_kperp2;

cureal Lx, Ly;

namespace{
    cureal *dv_rtmp;
    cucmplx *dv_ctmp;
}

/* ---------------------------------------------------------------- */

void init_four( void );
void finish_four( void );
__global__ void negative( cureal* );
__global__ void ddx( const cucmplx*, cucmplx* );
__global__ void neg_ddx( const cucmplx*, cucmplx* );
__global__ void ddy( const cucmplx*, cucmplx* );
__global__ void neg_ddy( const cucmplx*, cucmplx* );
__global__ void laplacian( const cucmplx*, cucmplx* );
__global__ void neg_lapinv( const cucmplx*, cucmplx * );
void get_vector( const cucmplx*, cureal*, cureal* );
void poisson_bracket( const cureal*, const cureal*, const cucmplx*, cureal* );
__global__ void mult_real_field( const cureal*, cureal* );
__global__ void add_real_field( const cureal*, cureal* );

/* ---------------------------------------------------------------- */

void init_four
    ( void 
){
    int ikx, iky;

    init_fft();

    kx = new cureal [nkx];
    for( ikx = 0; ikx < nkxh; ikx++ ) kx[ikx] = ikx * 2*M_PI/Lx;
    for( ikx = 0; ikx < nkxh-1; ikx++ ) kx[nkx-ikx-1] = -kx[ikx+1];

    ky = new cureal [nky];
    for( iky = 0; iky < nky; iky++ ) ky[iky] = iky * 2*M_PI/Ly;

    kperp2 = new cureal [nkx*nky];
    for( ikx = 0; ikx < nkx; ikx++ ){
        for( iky = 0; iky < nky; iky++ )
            kperp2[ikx*nky+iky] = kx[ikx]*kx[ikx] + ky[iky]*ky[iky];
    }

    cudaMalloc( (void**)&dv_kx, sizeof(cureal) * nkx );
    cudaMemcpy( dv_kx, kx, sizeof(cureal) * nkx, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_kx, &dv_kx, sizeof(dv_kx) );

    cudaMalloc( (void**)&dv_ky, sizeof(cureal) * nky );
    cudaMemcpy( dv_ky, ky, sizeof(cureal) * nky, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_ky, &dv_ky, sizeof(dv_ky) );

    cudaMalloc( (void**)&dv_kperp2, sizeof(cureal) * nkx * nky );
    cudaMemcpy( dv_kperp2, kperp2, sizeof(cureal) * nkx * nky, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_kperp2, &dv_kperp2, sizeof(dv_kperp2) );

    cudaMalloc( (void**)&dv_rtmp, sizeof(cureal) * nx * ny );
    cudaMalloc( (void**)&dv_ctmp, sizeof(cucmplx) * nkx * nky );
}

void finish_four
    ( void 
){
    finish_fft();

    delete[] kx;
    delete[] ky;
    delete[] kperp2;

    cudaFree( dv_kx );
    cudaFree( dv_ky );
    cudaFree( dv_kperp2 );
    cudaFree( dv_rtmp );
    cudaFree( dv_ctmp );
}

__global__ void negative
    ( cureal *field 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nx*ct_ny ) field[tid] = -field[tid];
}

__global__ void ddx
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid / ct_nky;

    if( tid < ct_nkx*ct_nky ) out[tid] = CIMAG * gb_kx[xid] * in[tid];
}

__global__ void neg_ddx
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid / ct_nky;

    if( tid < ct_nkx*ct_nky ) out[tid] = -CIMAG * gb_kx[xid] * in[tid];
}

__global__ void ddy
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = tid % ct_nky;

    if( tid < ct_nkx*ct_nky ) out[tid] = CIMAG * gb_ky[yid] * in[tid];
}

__global__ void neg_ddy
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = tid % ct_nky;

    if( tid > ct_nkx*ct_nky ) out[tid] = -CIMAG * gb_ky[yid] * in[tid];
}

__global__ void laplacian
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ) out[tid] = -gb_kperp2[tid] * in[tid];
}

__global__ void neg_lapinv
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if( tid <= ct_nkx*ct_nky-1 ) out[tid] = in[tid] / gb_kperp2[tid];
}

void get_vector
    ( const cucmplx *dv_aphi
    ,       cureal  *dv_vectx
    ,       cureal  *dv_vecty 
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    ddy <<< cgrid, block >>> ( dv_aphi, dv_ctmp );
    ktox( dv_ctmp, dv_vectx );

    neg_ddx <<< cgrid, block >>> ( dv_aphi, dv_ctmp );
    ktox( dv_ctmp, dv_vecty );
}

void poisson_bracket
    ( const cureal  *dv_vectx
    , const cureal  *dv_vecty
    , const cucmplx *in
    ,       cureal  *out
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );
    dim3 rgrid( (nx*ny+nthread-1)/ nthread );

    ddx <<< cgrid, block >>> ( in, dv_ctmp );
    ktox( dv_ctmp, out );
    mult_real_field <<< rgrid, block >>> ( dv_vectx, out );

    ddy <<< cgrid, block >>> ( in, dv_ctmp );
    ktox( dv_ctmp, dv_rtmp );
    mult_real_field <<< rgrid, block >>> ( dv_vecty, dv_rtmp );
    add_real_field <<< rgrid, block >>> ( dv_rtmp, out );
}

__global__ void mult_real_field
    ( const cureal *a
    ,       cureal *b 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nx*ct_ny ) b[tid] = a[tid] * b[tid];
}

__global__ void add_real_field
    ( const cureal *a
    ,       cureal *b 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nx*ct_ny ) b[tid] = a[tid] + b[tid];
}
