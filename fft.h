#ifndef __FFT_H__
#define __FFT_H__

extern int nthread;
extern int nx, ny, nkx, nky, nkxh, nkxh2, nkxpad, ncy;
extern __constant__ int ct_nx, ct_ny, ct_nkx, ct_nky;
extern __constant__ int ct_nkxh, ct_nkxh2, ct_nkxpad, ct_ncy;

void init_fft
    ( void
);

void finish_fft
    ( void
);

void xtok
    ( cureal  *in
    , cucmplx *out
);

void ktox
    ( const cucmplx *in
    ,       cureal  *out
);

void ktox_1d
    ( const cucmplx *in
    ,       cureal  *out
);

#endif
