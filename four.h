#ifndef __FOUR_H__
#define __FOUR_H__

extern cureal *kx, *ky, *kperp2, *dv_kx, *dv_ky, *dv_kperp2;
extern __device__ cureal *gb_kx, *gb_ky, *gb_kperp2;

extern cureal Lx, Ly;

extern void init_four( void );
extern void finish_four( void );
extern __global__ void negative( cureal* );
extern __global__ void ddx( const cucmplx*, cucmplx* );
extern __global__ void neg_ddx( const cucmplx*, cucmplx* );
extern __global__ void ddy( const cucmplx*, cucmplx* );
extern __global__ void neg_ddy( const cucmplx*, cucmplx* );
extern __global__ void laplacian( const cucmplx*, cucmplx* );
extern __global__ void neg_lapinv( const cucmplx*, cucmplx* );
extern void get_vector( const cucmplx*, cureal*, cureal* );
extern void poisson_bracket( const cureal*, const cureal*, const cucmplx*, cureal* );
extern __global__ void mult_real_field( const cureal*, cureal* );
extern __global__ void add_real_field( const cureal*, cureal* );

#endif
