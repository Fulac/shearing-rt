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

void file_out
    ( int    istep
    , cureal time
);

void en_spectral
    ( int    istep
    , cureal time
);

#endif
