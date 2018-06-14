#ifndef __FOUR_H__
#define __FOUR_H__

extern cureal *kx, *ky, *kperp2, *dv_kx, *dv_ky, *dv_kperp2;
extern __device__ cureal *gb_kx, *gb_ky, *gb_kperp2;

extern cureal Lx, Ly;

void init_four
    ( void
);

void finish_four
    ( void
);

__global__ void negative
    ( cureal *field
);

__global__ void ddx
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void neg_ddx
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void ddy
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void neg_ddy
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void laplacian
    ( const cucmplx *in
    ,       cucmplx *out
);

__global__ void neg_lapinv
    ( const cucmplx *in
    ,       cucmplx *out
);

void get_vector
    ( const cucmplx *aphi
    ,       cureal  *dv_vectx
    ,       cureal  *dv_vecty
);

void poisson_bracket
    ( const cureal  *dv_vectx
    , const cureal  *dv_vecty
    , const cucmplx *in
    ,       cureal  *out
);

__global__ void mult_real_field
    ( const cureal *fa
    ,       cureal *fb
);

__global__ void add_real_field
    ( const cureal *fa
    ,       cureal *fb
);

#endif
