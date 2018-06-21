#!/bin/bash

INITIAL_RHO="1e-2 1e-1"
KAPPAS="0"
RHO0_PRIME="1.0"

for INITIAL_RHO in $INITIAL_RHO; do
  for KAPPA in $KAPPAS; do
    FILE="alpha${RHO0_PRIME}_rho${INITIAL_RHO}_kappa${KAPPA}_nx256_ny512.ini"

    echo "[simulation]" > $FILE
    echo "nx = 256" >> $FILE
    echo "ny = 512" >> $FILE
    echo "" >> $FILE
    echo "eps = 1e-10" >> $FILE
    echo "time step = 1e-3" >> $FILE
    echo "time max = 30" >> $FILE
    echo "" >> $FILE
    echo "cuda thread num = 1024" >> $FILE
    echo "" >> $FILE
    echo "# M_PI" >> $FILE
    echo "Lx = 3.1415926535897932384626433832795" >> $FILE
    echo "Ly = 3.1415926535897932384626433832795" >> $FILE
    echo "" >> $FILE
    echo "[problem]" >> $FILE
    echo "initial noise = true" >> $FILE
    echo "nu = 1e-3" >> $FILE
    echo "kappa = $KAPPA" >> $FILE
    echo "" >> $FILE
    echo "sigma = 1.0" >> $FILE
    echo "g = 1.0" >> $FILE
    echo "rho0 = 1.0" >> $FILE
    echo "rho0_prime = $RHO0_PRIME" >> $FILE
    echo "" >> $FILE
    echo "rho_eps1 = $INITIAL_RHO" >> $FILE
    echo "rho_eps2 = 0.5e-2" >> $FILE
    echo "" >> $FILE
    echo "[output]" >> $FILE
    echo "write output = true" >> $FILE
    echo "output time step = 1.0" >> $FILE
    echo "output loop count = 100" >> $FILE
  done
done
