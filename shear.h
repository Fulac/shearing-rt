#ifndef __SHEAR_H__
#define __SHEAR_H__

extern __device__ cureal *gb_ky_shift, *gb_kperp2_shear;

void init_shear
    ( void
);

void finish_shear
    ( void
);

__global__ void ddy_shear
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void laplacian_shear
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void neg_lapinv_shear
    ( const cucmplx *in
    ,       cucmplx *out
);

void get_vector_shear
    ( const cucmplx *dv_aphi
    ,       cureal  *dv_vectx
    ,       cureal  *dv_vecty
);

void poisson_bracket_shear
    ( const cureal  *av_vectx
    , const cureal  *dv_vecty
    , const cucmplx *in
    ,       cureal  *out
);

__global__ void seq_ktox_shear
    ( const cucmplx *in
    ,       cureal  *out
);

void ktox_shear
    ( const cucmplx *in
    ,       cureal  *out
);

void update_shear
    ( const cureal delt
    , const cureal time
    , const int    istep
);

#endif
