#!/bin/bash

mkdir -p fields/omg
mkdir -p fields/phi
mkdir -p fields/rho
mkdir -p fields/data

mkdir -p ksre/omg
mkdir -p ksre/phi
mkdir -p ksre/rho
mkdir -p ksre/data

mkdir -p ksim/omg
mkdir -p ksim/phi
mkdir -p ksim/rho
mkdir -p ksim/data

mkdir -p kx_ensp/omg
mkdir -p kx_ensp/phi
mkdir -p kx_ensp/rho
mkdir -p kx_ensp/data

mkdir -p ky_ensp/omg
mkdir -p ky_ensp/phi
mkdir -p ky_ensp/rho
mkdir -p ky_ensp/data

mkdir -p kdata


if [[ -e n00000_t00.000000.dat ]]; then
  mv n*.dat fields/data
fi

if [[ -e ksre_t00.000000.dat ]]; then
  mv ksre*.dat ksre/data
fi
if [[ -e ksim_t00.000000.dat ]]; then
  mv ksim*.dat ksim/data
fi

if [[ -e kx_ensp.txt ]]; then
  mv kx_ensp.txt kx_ensp/data
  mv kx*.dat kx_ensp/data
fi
if [[ -e ky_ensp.txt ]]; then
  mv ky_ensp.txt ky_ensp/data
  mv ky*.dat ky_ensp/data
fi

if [[ -e befk_n00000_t00.000000.dat ]]; then
  mv bef*.dat kdata
fi
if [[ -e after_kdata.txt ]]; then
  rm after_kdata.txt
  mv aft*.dat kdata
fi


cd fields/data
if [[ -e n00000_t00.000000.dat ]]; then
  gnuplot ../../gp_files/plot_fields_omg.gp
  mv *.png ../omg

  gnuplot ../../gp_files/plot_fields_phi.gp
  mv *.png ../phi

  gnuplot ../../gp_files/plot_fields_rho.gp
  mv *.png ../rho
fi
cd ../..


cd ksre/data
if [[ -e ksre_t00.000000.dat ]]; then
  gnuplot ../../gp_files/plot_kspace_omg.gp
  mv *.png ../omg

  gnuplot ../../gp_files/plot_kspace_phi.gp
  mv *.png ../phi

  gnuplot ../../gp_files/plot_kspace_rho.gp
  mv *.png ../rho
fi
cd ../..

cd ksim/data
if [[ -e ksim_t00.000000.dat ]]; then
  gnuplot ../../gp_files/plot_kspace_omg.gp
  mv *.png ../omg

  gnuplot ../../gp_files/plot_kspace_phi.gp
  mv *.png ../phi

  gnuplot ../../gp_files/plot_kspace_rho.gp
  mv *.png ../rho
fi
cd ../..


cd kx_ensp/data
if [[ -e kx_ensp.txt ]]; then
  rm kx_ensp.txt

  gnuplot ../../gp_files/plot_kx_ensp_omg.gp
  mv *.eps ../omg

  gnuplot ../../gp_files/plot_kx_ensp_phi.gp
  mv *.eps ../phi

  gnuplot ../../gp_files/plot_kx_ensp_rho.gp
  mv *.eps ../rho
fi
cd ../..

cd ky_ensp/data
if [[ -e ky_ensp.txt ]]; then
  rm ky_ensp.txt

  gnuplot ../../gp_files/plot_ky_ensp_omg.gp
  mv *.eps ../omg

  gnuplot ../../gp_files/plot_ky_ensp_phi.gp
  mv *.eps ../phi

  gnuplot ../../gp_files/plot_ky_ensp_rho.gp
  mv *.eps ../rho
fi
cd ../..
