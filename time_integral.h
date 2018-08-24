#ifndef __TINT_H__
#define __TINT_H__

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

cureal maxvalue_search
    ( cureal *dv_field
);

void time_advance
    ( int    &istep
    , cureal &time
);

#endif
