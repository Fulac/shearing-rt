// CUDA headers
#include <cuda.h>

// Local headers
#include "cmplx.h"
#include "fft.h"

/* ---------------------------------------------------------------------------------------------- */
/*  Global Variables Definition                                                                   */
/* ---------------------------------------------------------------------------------------------- */

// プログラム全体で使用する変数を定義
cureal delt, tmax;
cureal nu, kappa, sigma, rho0, rho0_prime, g, rho_eps1, rho_eps2, eps;
__constant__ cureal ct_nu, ct_sigma, ct_rho0, ct_rho0_prime, ct_g, ct_kappa, ct_rho_eps2;
cureal dx, dy;

cureal *xx, *yy, *omgz, *phi, *rho;
cureal *dv_xx, *dv_yy, *dv_omgz, *dv_phi, *dv_rho, *dv_vx, *dv_vy;
__device__ cureal *gb_xx, *gb_yy;
cucmplx *dv_aomg0, *dv_aomg1, *dv_aomg2;
cucmplx *dv_domg0, *dv_domg1, *dv_domg2;
cucmplx *dv_arho0, *dv_arho1, *dv_arho2;
cucmplx *dv_drho0, *dv_drho1, *dv_drho2;
cucmplx *dv_aphi;

////////////////////////////////////////////////////////////////////////////////////////////////////

/* ---------------------------------------------------------------------------------------------- */
/*  Function Prototype                                                                            */
/* ---------------------------------------------------------------------------------------------- */

void init_fields
    ( void
);

void finish_fields
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void allocate_CPU
    ( void
);

static void deallocate_CPU
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void allocate_GPU
    ( void
);

static void deallocate_GPU
    ( void
);

////////////////////////////////////////////////////////////////////////////////////////////////////


/* ---------------------------------------------------------------------------------------------- */
/*  Function Definition                                                                           */
/* ---------------------------------------------------------------------------------------------- */

void init_fields
    ( void 
){
    cudaMemcpyToSymbol( ct_nu,         &nu,         sizeof(cureal) );
    cudaMemcpyToSymbol( ct_sigma,      &sigma,      sizeof(cureal) );
    cudaMemcpyToSymbol( ct_rho0,       &rho0,       sizeof(cureal) );
    cudaMemcpyToSymbol( ct_rho0_prime, &rho0_prime, sizeof(cureal) );
    cudaMemcpyToSymbol( ct_g,          &g,          sizeof(cureal) );
    cudaMemcpyToSymbol( ct_kappa,      &kappa,      sizeof(cureal) );
    cudaMemcpyToSymbol( ct_rho_eps2,   &rho_eps2,   sizeof(cureal) );

    allocate_CPU();
    allocate_GPU();
}

void finish_fields
    ( void 
){
    deallocate_CPU();
    deallocate_GPU();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void allocate_CPU
    ( void 
){
    int nn = nx * ny;

    xx = new cureal [nx+1];
    yy = new cureal [ny+1];
    omgz = new cureal [nn];
    phi = new cureal [nn];
    rho = new cureal [nn];
}

static void deallocate_CPU
    ( void 
){
    delete[] xx;
    delete[] yy;
    delete[] omgz;
    delete[] phi;
    delete[] rho;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void allocate_GPU
    ( void 
){
    int nn;

    cudaMalloc( (void**)&dv_xx, sizeof(cureal) * (nx+1) );
    cudaMalloc( (void**)&dv_yy, sizeof(cureal) * (ny+1) );

    nn = nx * ny;
    cudaMalloc( (void**)&dv_vx, sizeof(cureal) * nn );
    cudaMemset( dv_vx, 0, sizeof(cureal) * nn );
    cudaMalloc( (void**)&dv_vy, sizeof(cureal) * nn );
    cudaMemset( dv_vy, 0, sizeof(cureal) * nn );
    cudaMalloc( (void**)&dv_omgz, sizeof(cureal) * nn );
    cudaMalloc( (void**)&dv_phi, sizeof(cureal) * nn );
    cudaMalloc( (void**)&dv_rho, sizeof(cureal) * nn );

    nn = nkx * nky;
    cudaMalloc( (void**)&dv_aomg0, sizeof(cucmplx) * nn );
    cudaMemset( dv_aomg0, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_aomg1, sizeof(cucmplx) * nn );
    cudaMemset( dv_aomg1, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_aomg2, sizeof(cucmplx) * nn );
    cudaMemset( dv_aomg2, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_domg0, sizeof(cucmplx) * nn );
    cudaMemset( dv_domg0, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_domg1, sizeof(cucmplx) * nn );
    cudaMemset( dv_domg1, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_domg2, sizeof(cucmplx) * nn );
    cudaMemset( dv_domg2, 0, sizeof(cucmplx) * nn );

    cudaMalloc( (void**)&dv_arho0, sizeof(cucmplx) * nn );
    cudaMemset( dv_aomg0, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_arho1, sizeof(cucmplx) * nn );
    cudaMemset( dv_aomg1, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_arho2, sizeof(cucmplx) * nn );
    cudaMemset( dv_aomg2, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_drho0, sizeof(cucmplx) * nn );
    cudaMemset( dv_domg0, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_drho1, sizeof(cucmplx) * nn );
    cudaMemset( dv_domg1, 0, sizeof(cucmplx) * nn );
    cudaMalloc( (void**)&dv_drho2, sizeof(cucmplx) * nn );
    cudaMemset( dv_domg2, 0, sizeof(cucmplx) * nn );

    cudaMalloc( (void**)&dv_aphi, sizeof(cucmplx) * nn );
    cudaMemset( dv_aphi, 0, sizeof(cucmplx) * nn );
}

static void deallocate_GPU
    ( void 
){
    cudaFree( dv_xx );
    cudaFree( dv_yy );
    cudaFree( dv_vx );
    cudaFree( dv_vy );
    cudaFree( dv_omgz );
    cudaFree( dv_phi );
    cudaFree( dv_rho );


    cudaFree( dv_aomg0 );
    cudaFree( dv_aomg1 );
    cudaFree( dv_aomg2 );
    cudaFree( dv_domg0 );
    cudaFree( dv_domg1 );
    cudaFree( dv_domg2 );

    cudaFree( dv_arho0 );
    cudaFree( dv_arho1 );
    cudaFree( dv_arho2 );
    cudaFree( dv_drho0 );
    cudaFree( dv_drho1 );
    cudaFree( dv_drho2 );

    cudaFree( dv_aphi );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
