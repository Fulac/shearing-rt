#ifndef __OUTPUT_H__
#define __OUTPUT_H__

extern bool   write_fields;
extern int    nwrite;
extern cureal output_time, next_output_time;

void init_output
    ( void
);

void finish_output
    ( void
);

void input_data
    ( void
);

void output_fields
    ( const int    istep
    , const cureal time
);

void output_maxamp
    ( const cureal time
);

void en_spectral
    ( const int    istep
    , const cureal time
);

void k_data_bef
    ( const cureal time
    , const int    istep
);

void k_data_aft
    ( const cureal time
    , const int    istep
);

#endif
