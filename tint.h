#ifndef __TINT_H__
#define __TINT_H__

extern int nrst, nst;
extern bool noise;

extern void init_tint( void );
extern void finish_tint( void );
extern void initialize( void );
extern void time_advance( int&, cureal& );

#endif
