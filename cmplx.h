#ifndef __CMPLX_H__
#define __CMPLX_H__

// CUDA headers
#include <cuComplex.h>

#ifdef DBLE
    typedef double cureal;
    typedef cuDoubleComplex cucmplx;
#else
    typedef float cureal;
    typedef cuComplex cucmplx;
#endif

#ifdef DBLE
    #define CIMAG ( make_cuDoubleComplex(0.0, 1.0) )
    #define CZERO ( make_cuDoubleComplex(0.0, 0.0) )
#else
    #define CIMAG ( make_cuComplex(0.0f, 1.0f) )
    #define CZERO ( make_cuComplex(0.0f, 0.0f) )
#endif

__host__ __device__ static __inline__ cucmplx cmplx
    ( const cureal a
    , const cureal b
){
    #ifdef DBLE
        return make_cuDoubleComplex( a, b );
    #else
        return make_cuComplex( a, b );
    #endif
}

__host__ __device__ __inline__ cucmplx operator+
    ( const cucmplx a
    , const cucmplx b
){
    #ifdef DBLE
        return cuCadd( a, b );
    #else
        return cuCaddf( a, b );
    #endif
}

__host__ __device__ __inline__ cucmplx operator+
    ( const cucmplx a
    , const cureal  b
){
    return cmplx( a.x + b, a.y );
}

__host__ __device__ __inline__ cucmplx operator+
    ( const cureal  a
    , const cucmplx b
){
    return cmplx( a + b.x, b.y );
}

__host__ __device__ __inline__ cucmplx operator-
    ( const cucmplx a
    , const cucmplx b
){
    #ifdef DBLE
        return cuCsub( a, b );
    #else
        return cuCsubf( a, b );
    #endif
}

__host__ __device__ __inline__ cucmplx operator-
    ( const cucmplx a
    , const cureal  b
){
    return cmplx( a.x - b, a.y );
}

__host__ __device__ __inline__ cucmplx operator-
    ( const cureal  a
    , const cucmplx b
){
    return cmplx( a - b.x, -b.y );
}

__host__ __device__ __inline__ cucmplx operator-
    ( const cucmplx a
){
    return cmplx( -a.x, -a.y );
}

__host__ __device__ __inline__ cucmplx operator*
    ( const cucmplx a
    , const cucmplx b
){
    #ifdef DBLE
        return cuCmul( a, b );
    #else
        return cuCmulf( a, b );
    #endif
}

__host__ __device__ __inline__ cucmplx operator*
    ( const cucmplx a
    , const cureal  b
){
    return cmplx( a.x * b, a.y * b );
}

__host__ __device__ __inline__ cucmplx operator*
    ( const cureal  a
    , const cucmplx b
){
    return cmplx( a * b.x, a * b.y );
}

__host__ __device__ __inline__ cucmplx operator/
    ( const cucmplx a
    , const cucmplx b
){
    #ifdef DBLE
        return cuCdiv( a, b );
    #else
        return cuCdivf( a, b );
    #endif
}

__host__ __device__ __inline__ cucmplx operator/
    ( const cureal  a
    , const cucmplx b
){
    cureal denom = b.x * b.x + b.y * b.y;
    return cmplx( a * b.x / denom, -a * b.y / denom );
}

__host__ __device__ __inline__ cucmplx operator/
    ( const cucmplx a
    , const cureal  b
){
    return cmplx( a.x / b, a.y / b );
}

#endif
