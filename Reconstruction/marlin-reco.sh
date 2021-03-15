#!/bin/bash


for (( ; ; ))
do
   
   if [ -f rec.lock ]; then
        ## prepare for reconstruction
        export BASE=/home/ilc/iLCInstall/ilcsoft
        source /home/ilc/ilcsoft/v02-01-pre/init_ilcsoft.sh
        git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git
        
        cp $BASE/AHcalDigi.xml $BASE/ILDConfig/StandardConfig/production/CaloDigi/AHcalDigi.xml
        cd ILDConfig/StandardConfig/production

        export REC_MODEL=ILD_l5_o1_v02
        export FILE=$1

        ##
        ## Full reconstruction of showers with
        ##
        echo "-- Running Reco ${REC_MODEL} ..."
        Marlin MarlinStdReco.xml \
        --constant.lcgeo_DIR=$lcgeo_DIR \
        --constant.DetectorModel=${REC_MODEL} \
        --constant.OutputBaseName=${BASE}/${FILE} \
        --constant.RunBeamCalReco=false \
        --global.LCIOInputFiles=${BASE}/${FILE}

        cd $BASE
        rm rec.lock
        break;
   else 
      echo "waiting for input file for reconstruction"
      sleep 20;
   fi

done



