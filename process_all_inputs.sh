#!/bin/bash

set -e

SIMEXE=$(readlink -f shrt)

if [[ "$#" = 2 ]]; then
   pushd . >/dev/null 2>&1
   cd $1
   INPUTDIR=$PWD
   popd >/dev/null 2>&1
   
   if [[ -d "$2" ]]; then
      read -p "Directory ${2} already exists; continue (y/n)?" -n 1 -r
      echo
      if [[ $REPLY =~ ^[^Yy]$ ]]; then
         echo "Aborted."
         exit 1
      fi
   else
      mkdir -p $2
   fi
   
   pushd . >/dev/null 2>&1
   cd $2
   DATADIR=$PWD
   popd >/dev/null 2>&1
else
   echo "usage: $0 source_dir data_dir exe_file"
   exit 1
fi

LOGFILE="${DATADIR}/process.log"

echo "$(date +%T) :: Started to process directory '${INPUTDIR}'" | tee $LOGFILE

cd "$INPUTDIR"

for INPUT in *.ini; do
   echo "$(date '+%F~%T') :: --- Processing ${INPUT}..." | tee -a $LOGFILE
   
   SIMNAME=$(basename "${INPUT}" .txt)
   SIMDIR="${DATADIR}/${SIMNAME}"
   
   mkdir -p "${SIMDIR}"
   cp "$INPUT" "${SIMDIR}/config.ini"
   
   cd "$SIMDIR"
   $SIMEXE | tee "${INPUT}.log"
   cd "$INPUTDIR"
done

echo "$(date +%T) :: Finished." | tee -a $LOGFILE

