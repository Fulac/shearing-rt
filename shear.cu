#include <cstdio>
#include <cmath>
#include <cuda.h>
#include "cmplx.h"
#include "fft.h"
#include "four.h"
#include "fields.h"

namespace{
    int *ikx_indexed, *dv_ikx_indexed, jump_flag, *dv_jump;
    __device__ int *gb_ikx_indexed, *gb_jump;
    cureal *dv_ky_shift, *kperp2_shear, *dv_kperp2_shear;

    cureal  *dv_rtmp;
    cucmplx *dv_ctmp1, *dv_ctmp2, *dv_ctmp3;
}

__device__ cureal *gb_ky_shift, *gb_kperp2_shear;

/* ---------------------------------------------------------------- */

void init_shear( void );
void finish_shear( void );

__global__ void ddy_shear( const cucmplx*, cucmplx* );
__global__ void laplacian_shear( const cucmplx*, cucmplx* );
__global__ void neg_lapinv_shear( const cucmplx*, cucmplx* );
void get_vector_shear( const cucmplx*, cureal*, cureal* );
void poisson_bracket_shear( const cureal*, const cureal*, const cucmplx*, cureal* );

__global__ void seq_ktox_shear( const cucmplx*, cureal* );
void ktox_shear( const cucmplx*, cureal* );
__global__ static void idft_shear_y( const cucmplx*, cucmplx* );
__global__ static void pad2d( const cucmplx*, cucmplx* );

void update_shear( const cureal );
__global__ static void shearing_ky( const cureal );
__global__ static void get_kperp2_shear( void );
__global__ static void shearing_field( const cucmplx*, cucmplx* );

/* ---------------------------------------------------------------- */

void init_shear
    ( void
){
    kperp2_shear = new cureal [nkx*nky];

    cudaMalloc( (void**)&dv_kperp2_shear, sizeof(cureal)*nkx*nky );
    cudaMemcpy( dv_kperp2_shear, kperp2, sizeof(cureal)*nkx*nky, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_kperp2_shear, &dv_kperp2_shear, sizeof(dv_kperp2_shear) );

    ikx_indexed = new int [nkx];
    ikx_indexed[0] = 0;
    for( int ikx = 1; ikx < nkx; ikx++ ) ikx_indexed[ikx] = nkx - ikx;

    cudaMalloc( (void**)&dv_ikx_indexed, sizeof(int) * nkx );
    cudaMemcpy( dv_ikx_indexed, ikx_indexed, sizeof(int) * nkx, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( gb_ikx_indexed, &dv_ikx_indexed, sizeof(dv_ikx_indexed) );

    cudaMalloc( (void**)&dv_jump, sizeof(int)*nkx );
    cudaMemset( dv_jump, 0, sizeof(int)*nkx );
    cudaMemcpyToSymbol( gb_jump, &dv_jump, sizeof(dv_jump) );

    cudaMalloc( (void**)&dv_ky_shift, sizeof(cureal)*nkx );
    cudaMemset( dv_ky_shift, 0, sizeof(cureal)*nkx );
    cudaMemcpyToSymbol( gb_ky_shift, &dv_ky_shift, sizeof(dv_ky_shift) );

    cudaMalloc( (void**)&dv_rtmp,  sizeof(cureal)*nx*ny );
    cudaMalloc( (void**)&dv_ctmp1, sizeof(cucmplx)*nkx*nky );
    cudaMalloc( (void**)&dv_ctmp2, sizeof(cucmplx)*nkx*ny );
    cudaMalloc( (void**)&dv_ctmp3, sizeof(cucmplx)*nx*ny );
}

void finish_shear
    ( void
){
    delete[] kperp2_shear;
    delete[] ikx_indexed;

    cudaFree( dv_jump );
    cudaFree( dv_ky_shift );
    cudaFree( dv_kperp2_shear );
    cudaFree( dv_rtmp );
    cudaFree( dv_ctmp1 );
    cudaFree( dv_ctmp2 );
    cudaFree( dv_ctmp3 );
}

__global__ void ddy_shear
    ( const cucmplx *in
    ,       cucmplx *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_nky, yid = tid%ct_nky;
    cureal kys;

    if( tid < ct_nkx*ct_nky ){
        kys = gb_ky[yid] + gb_ky_shift[xid];
        out[tid] = CIMAG * kys * in[tid];
    }
}

__global__ void laplacian_shear
    ( const cucmplx *in
    ,       cucmplx *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < ct_nkx*ct_nky ) out[tid] = -gb_kperp2_shear[tid] * in[tid];
}

__global__ void neg_lapinv_shear
    ( const cucmplx *in
    ,       cucmplx *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if( tid <= ct_nkx*ct_nky-1 ) out[tid] = in[tid] / gb_kperp2_shear[tid];
}

void get_vector_shear
    ( const cucmplx *dv_aphi
    ,       cureal  *dv_vectx
    ,       cureal  *dv_vecty
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    ddy_shear <<< cgrid, block >>> ( dv_aphi, dv_ctmp1 );
    ktox( dv_ctmp1, dv_vectx );

    neg_ddx <<< cgrid, block >>> ( dv_aphi, dv_ctmp1 );
    ktox( dv_ctmp1, dv_vecty );
}

void poisson_bracket_shear
    ( const cureal  *dv_vectx
    , const cureal  *dv_vecty
    , const cucmplx *in
    ,       cureal  *out
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    ddx <<< cgrid, block >>> ( in, dv_ctmp1 );
    ktox( dv_ctmp1, out );
    mult_real_field <<< rgrid, block >>> ( dv_vectx, out );

    ddy_shear <<< cgrid, block >>> ( in, dv_ctmp1 );
    ktox( dv_ctmp1, dv_rtmp );
    mult_real_field <<< rgrid, block >>> ( dv_vecty, dv_rtmp );

    add_real_field <<< rgrid, block >>> ( dv_rtmp, out );
}

__global__ void seq_ktox_shear
    ( const cucmplx *in
    ,       cureal  *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_ny, yid = tid%ct_ny;

    if( tid < ct_nx*ct_ny ){
        cureal kx, ky;
        cureal out_data = 0;

        for( int ikx = 0; ikx < ct_nkx; ikx++ ){
            kx = gb_kx[ikx];
            ky = gb_ky[0] + gb_ky_shift[ikx];
            out_data += in[ikx*ct_nky].x*cos(kx*gb_xx[xid]+ky*gb_yy[yid])
                      - in[ikx*ct_nky].y*sin(kx*gb_xx[xid]+ky*gb_yy[yid]);

            for( int iky = 1; iky < ct_nky; iky++ ){
                ky = gb_ky[iky] + gb_ky_shift[ikx];
                out_data += 2*in[ikx*ct_nky+iky].x*cos(kx*gb_xx[xid]+ky*gb_yy[yid])
                          - 2*in[ikx*ct_nky+iky].y*sin(kx*gb_xx[xid]+ky*gb_yy[yid]);
            }
        }
        out[tid] = out_data;
    }
}

void ktox_shear
    ( const cucmplx *in
    ,       cureal  *out
){
    dim3 block( nthread );
    dim3 rcgrid( (nkx*ny+nthread-1)/nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );

    idft_shear_y <<< rcgrid, block >>> ( in, dv_ctmp2 );
    pad2d <<< rgrid, block >>> ( dv_ctmp2, dv_ctmp3 );
    ktox_1d( dv_ctmp3, out );
}

__global__ static void idft_shear_y
    ( const cucmplx *in
    ,       cucmplx *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_ny, yid = tid%ct_ny;

    if( tid < ct_nkx*ct_ny ){
        cucmplx out_data = CZERO;
        cureal  y_data   = gb_yy[yid];
        cureal  ky;
        cureal  ky_shift = gb_ky_shift[xid];

        ky = gb_ky[0] + ky_shift;

        out_data.x = in[xid*ct_nky].x * cos( ky * y_data )
                   - in[xid*ct_nky].y * sin( ky * y_data );

        out_data.y = in[xid*ct_nky].y * cos( ky * y_data )
                   + in[xid*ct_nky].x * sin( ky * y_data );

        for( int iky = 1; iky < ct_nky; iky++ ){
            ky = gb_ky[iky] + ky_shift;

            out_data.x += 2 * in[xid*ct_nky+iky].x * cos( ky * y_data )
                        - 2 * in[xid*ct_nky+iky].y * sin( ky * y_data );

            out_data.y += 2 * in[xid*ct_nky+iky].y * cos( ky * y_data )
                        + 2 * in[xid*ct_nky+iky].x * sin( ky * y_data );
        }

        out[tid] = out_data;
    }
}

__global__ static void pad2d
    ( const cucmplx *in
    ,       cucmplx *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_ny, yid = tid%ct_ny;

    if( tid < ct_nx*ct_ny ){
        if( xid < ct_nkxh )       out[tid] = in[xid*ct_ny+yid];
        else if( xid > ct_nkxh2 ) out[tid] = in[(xid-ct_nkxpad)*ct_ny+yid];
        else                      out[tid] = CZERO;
    }
}

void update_shear
    ( const cureal delt 
){
    dim3 block( nthread );
    dim3 kxgrid( ((nkx-1)+nthread-1)/nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    shearing_ky <<< kxgrid, block >>> ( delt );
    get_kperp2_shear <<< cgrid, block >>> ();

    cudaMemcpy( &jump_flag, dv_jump+1, sizeof(int), cudaMemcpyDeviceToHost );
    if( jump_flag != 0 ){
        cudaMemcpy( dv_ctmp1, dv_aomg0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_aomg0 );
        cudaMemcpy( dv_ctmp1, dv_aomg1, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_aomg1 );
        cudaMemcpy( dv_ctmp1, dv_aomg2, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_aomg2 );

        cudaMemcpy( dv_ctmp1, dv_domg0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_domg0 );
        cudaMemcpy( dv_ctmp1, dv_domg1, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_domg1 );
        cudaMemcpy( dv_ctmp1, dv_domg2, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_domg2 );

        cudaMemcpy( dv_ctmp1, dv_arho0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_arho0 );
        cudaMemcpy( dv_ctmp1, dv_arho1, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_arho1 );
        cudaMemcpy( dv_ctmp1, dv_arho2, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_arho2 );

        cudaMemcpy( dv_ctmp1, dv_drho0, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_drho0 );
        cudaMemcpy( dv_ctmp1, dv_drho1, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_drho1 );
        cudaMemcpy( dv_ctmp1, dv_drho2, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_drho2 );

        cudaMemcpy( dv_ctmp1, dv_aphi, sizeof(cucmplx)*nkx*nky, cudaMemcpyDeviceToDevice );
        shearing_field <<< cgrid, block >>> ( dv_ctmp1, dv_aphi );
    }
}

__global__ static void shearing_ky
    ( const cureal delt
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if( tid <= ct_nkx-1 ){
        gb_ky_shift[tid] = gb_ky_shift[tid] - ct_sigma * gb_kx[tid] * delt;
        gb_jump[tid] = floor( gb_ky_shift[tid]/gb_ky[1] + 0.5 );

        __syncthreads();

        if( gb_jump[1] != 0 )
            gb_ky_shift[tid] = gb_ky_shift[tid] - gb_jump[tid] * gb_ky[1];
    }
}

__global__ static void get_kperp2_shear
    ( void
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_nky, yid = tid%ct_nky;
    cureal kxs, kys;

    if( tid < ct_nkx*ct_nky ){
        kxs = gb_kx[xid];
        kys = gb_ky[yid] + gb_ky_shift[xid];
        gb_kperp2_shear[tid] = kxs*kxs + kys*kys;
    }
}

__global__ static void shearing_field
    ( const cucmplx *in
    ,       cucmplx *out
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_nky, yid = tid%ct_nky;

    if( tid < ct_nkx*ct_nky ){
        if( gb_jump[xid] <= 0 ){
            if( yid-gb_jump[xid] < ct_nky ){
                out[tid] = in[xid*ct_nky+(yid-gb_jump[xid])];
            }
            else{
                out[tid] = CZERO;
            }
        }
        else{
            if( yid-gb_jump[xid] > 0 ){
                out[tid] = in[xid*ct_nky+(yid-gb_jump[xid])];
            }
            else{
                out[tid].x =  in[gb_ikx_indexed[xid]*ct_nky+(gb_jump[xid]-yid)].x;
                out[tid].y = -in[gb_ikx_indexed[xid]*ct_nky+(gb_jump[xid]-yid)].y;
            }
        }
    }
}
