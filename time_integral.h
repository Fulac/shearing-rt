#ifndef __TINT_H__
#define __TINT_H__

extern int nrst, nst;
extern bool noise_flag;

void init_tint
    ( void
);

void finish_tint
    ( void
);

void initialize
    ( void
);

void time_advance
    ( int    &istep
    , cureal &time
);

#endif
