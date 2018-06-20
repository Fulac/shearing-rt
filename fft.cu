// CUDA headers
#include <cuda.h>
#include <cufft.h>

// Local headers
#include "cmplx.h"

/* ---------------------------------------------------------------------------------------------- */
/*  Global Variables Definition                                                                   */
/* ---------------------------------------------------------------------------------------------- */

// プログラム全体で使用する変数を定義
int nthread;
int nx, ny, nkx, nky, nkxh, nkxh2, nkxpad, ncy;
__constant__ int ct_nx, ct_ny, ct_nkx, ct_nky;
__constant__ int ct_nkxh, ct_nkxh2, ct_nkxpad, ct_ncy;

// このファイル内でのみ使用するグローバル変数を定義
namespace{
    cureal  *dv_rtmp;
    cucmplx *dv_ctmp1, *dv_ctmp2;

    cufftHandle pr2c, pc2r, pc2c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Prototype                                                                            */
/* ---------------------------------------------------------------------------------------------- */

void init_fft
    ( void
);
void finish_fft
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void xtok
    ( cureal  *in
    , cucmplx *out
);

__global__ static void scale_dealias
    ( const cucmplx *in
    ,       cucmplx *out
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void ktox
    ( const cucmplx *in
    ,       cureal  *out
);

__global__ static void pad2d
    ( const cucmplx *in
    ,       cucmplx *out
);

////////////////////////////////////////////////////////////////////////////////////////////////////

void ktox_1d
    ( const cucmplx *in
    ,       cureal  *out
);

__global__ static void transpose
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ static void trans_inv
    ( const cucmplx *in
    ,       cureal  *out
);

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Definition                                                                           */
/* ---------------------------------------------------------------------------------------------- */

void init_fft
    ( void 
){
    nkx    = (nx-1)/3*2 + 1;
    nky    = (ny-1)/3 + 1;
    nkxh   = (nkx+1)/2;
    nkxh2  = nx - nkxh;
    nkxpad = nx - nkxh*2 + 1;
    ncy    = ny/2 + 1;

    cudaMemcpyToSymbol( ct_nx,     &nx,     sizeof(int) );
    cudaMemcpyToSymbol( ct_ny,     &ny,     sizeof(int) );
    cudaMemcpyToSymbol( ct_nkx,    &nkx,    sizeof(int) );
    cudaMemcpyToSymbol( ct_nky,    &nky,    sizeof(int) );
    cudaMemcpyToSymbol( ct_nkxh,   &nkxh,   sizeof(int) );
    cudaMemcpyToSymbol( ct_nkxh2,  &nkxh2,  sizeof(int) );
    cudaMemcpyToSymbol( ct_nkxpad, &nkxpad, sizeof(int) );
    cudaMemcpyToSymbol( ct_ncy,    &ncy,    sizeof(int) );

    cudaMalloc( (void**)&dv_rtmp,  sizeof(cureal)*nx*ny   );
    cudaMalloc( (void**)&dv_ctmp1, sizeof(cucmplx)*nx*ncy );
    cudaMalloc( (void**)&dv_ctmp2, sizeof(cucmplx)*nx*ny );

    #ifdef DBLE
        cufftPlan2d( &pr2c, nx, ny, CUFFT_D2Z );
        cufftPlan2d( &pc2r, nx, ny, CUFFT_Z2D );
        cufftPlan1d( &pc2c, nx, CUFFT_Z2Z, 1 );
    #else
        cufftPlan2d( &pr2c, nx, ny, CUFFT_R2C );
        cufftPlan2d( &pc2r, nx, ny, CUFFT_C2R );
        cufftPlan1d( &pc2c, nx, CUFFT_C2C, 1 );
    #endif
}

void finish_fft
    ( void 
){
    cufftDestroy( pr2c );
    cufftDestroy( pc2r );
    cufftDestroy( pc2c );

    cudaFree( dv_rtmp );
    cudaFree( dv_ctmp1 );
    cudaFree( dv_ctmp2 );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void xtok
    ( cureal  *in
    , cucmplx *out 
){
    dim3 block( nthread );
    dim3 cgrid( (nkx*nky+nthread-1)/nthread );

    #ifdef DBLE
        cufftExecD2Z( pr2c, in, dv_ctmp1 );
    #else
        cufftExecR2C( pr2c, in, dv_ctmp1 );
    #endif

    scale_dealias <<< cgrid, block >>> ( dv_ctmp1, out );
}

__global__ static void scale_dealias
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_nky, yid = tid%ct_nky;

    if( tid < ct_nkx*ct_nky ){
        if( xid < ct_nkxh ) out[tid] = in[xid*ct_ncy+yid] / (ct_nx*ct_ny);
        else out[tid] = in[(xid+ct_nkxpad)*ct_ncy+yid] / (ct_nx*ct_ny);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ktox
    ( const cucmplx *in
    ,       cureal  *out 
){
    dim3 block( nthread );
    dim3 rcgrid( (nx*ncy+nthread-1)/nthread );

    pad2d <<< rcgrid, block >>> ( in, dv_ctmp1 );

    #ifdef DBLE
        cufftExecZ2D( pc2r, dv_ctmp1, out );
    #else
        cufftExecC2R( pc2r, dv_ctmp1, out );
    #endif
}

__global__ static void pad2d
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_ncy, yid = tid%ct_ncy;

    if( tid < ct_nx*ct_ncy ){
        if( yid < ct_nky ){
            if( xid < ct_nkxh ) out[tid] = in[xid*ct_nky+yid];
            else if( xid > ct_nkxh2 ) out[tid] = in[(xid-ct_nkxpad)*ct_nky+yid];
            else out[tid] = CZERO;
        }
        else out[tid] = CZERO;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ktox_1d
    ( const cucmplx *in
    ,       cureal  *out
){
    dim3 block( nthread );
    dim3 rgrid( (nx*ny+nthread-1)/nthread );

    transpose <<< rgrid, block >>> ( in, dv_ctmp2 );

    for( int iy = 0; iy < ny; iy++ ){
        #ifdef DBLE
            cufftExecZ2Z( pc2c, dv_ctmp2+iy*nx, dv_ctmp2+iy*nx, CUFFT_INVERSE );
        #else
            cufftExecC2C( pc2c, dv_ctmp2+iy*nx, dv_ctmp2+iy*nx, CUFFT_INVERSE );
        #endif
    }

    trans_inv <<< rgrid, block >>> ( dv_ctmp2, out );
}

__global__ static void transpose
    ( const cucmplx *in
    ,       cucmplx *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid/ct_ny, yid = tid%ct_ny;

    if( tid < ct_nx*ct_ny ){
        out[yid*ct_nx+xid] = in[tid];
    }
}

__global__ static void trans_inv
    ( const cucmplx *in
    ,       cureal  *out 
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int xid = tid%ct_nx, yid = tid/ct_nx;

    if( tid < ct_nx*ct_ny ){
        out[xid*ct_ny+yid] = in[tid].x;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
