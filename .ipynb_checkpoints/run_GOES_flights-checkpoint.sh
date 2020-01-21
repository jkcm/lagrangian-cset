#!/bin/csh
#PBS -l nodes=1:ppn=8:e5645
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -l walltime=80:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N GOES_Extract_fp

cd /home/disk/p/jkcm/Code/Lagrangian_CSET
/home/disk/p/jkcm/anaconda2/bin/python flightpath_GOES_extractor.py >>& flightpath_GOES_extractor.log
exit 0
