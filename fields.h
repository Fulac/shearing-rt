#ifndef __FIELDS_H__
#define __FIELDS_H__

extern cureal delt, tmax;
extern cureal cfl_vx, cfl_vy;
extern cureal nu, kappa, sigma, rho0, rho0_prime, g, rho_eps1, rho_eps2, eps;
extern __constant__ cureal ct_nu, ct_rho0, ct_rho0_prime, ct_g, ct_sigma, ct_kappa, ct_rho_eps2;
extern cureal dx, dy;

extern cureal *xx, *yy, *omgz, *phi, *rho;
extern cureal *dv_xx, *dv_yy, *dv_omgz, *dv_phi, *dv_rho, *dv_vx, *dv_vy;
extern __device__ cureal *gb_xx, *gb_yy;
extern cucmplx *dv_aomg0, *dv_aomg1, *dv_aomg2;
extern cucmplx *dv_domg0, *dv_domg1, *dv_domg2;
extern cucmplx *dv_arho0, *dv_arho1, *dv_arho2;
extern cucmplx *dv_drho0, *dv_drho1, *dv_drho2;
extern cucmplx *dv_aphi;

void init_fields
    ( void
);

void finish_fields
    ( void
);

#endif
