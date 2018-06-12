#ifndef __OUTPUT_H__
#define __OUTPUT_H__

extern bool write_fields;
extern int nwrite;
extern cureal otime, next_otime;

extern void init_output();
extern void finish_output();
extern void file_out( int, cureal );
extern void en_spectral( int, cureal );

#endif
