#ifndef __OUTPUT_H__
#define __OUTPUT_H__

extern bool   write_fields;
extern int    nwrite;
extern cureal otime, next_otime;

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
    ( int    istep
    , cureal time
);

void en_spectral
    ( int    istep
    , cureal time
);

void k_data
    ( int flag
);

#endif
