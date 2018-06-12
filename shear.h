#ifndef __SHEAR_H__
#define __SHEAR_H__

extern __device__ cureal *gb_ky_shift, *gb_kperp2_shear;

extern void init_shear( void );
extern void finish_shear( void );
extern __global__ void ddy_shear( const cucmplx*, cucmplx* );
extern __global__ void laplacian_shear( const cucmplx*, cucmplx* );
extern __global__ void neg_lapinv_shear( const cucmplx*, cucmplx* );
extern void get_vector_shear( const cucmplx*, cureal*, cureal* );
extern void poisson_bracket_shear( const cureal*, const cureal*, const cucmplx*, cureal* );
extern __global__ void seq_ktox_shear( const cucmplx*, cureal* );
extern void ktox_shear( const cucmplx*, cureal* );
extern void update_shear( const cureal );

#endif
