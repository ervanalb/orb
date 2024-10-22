#!/bin/bash

export GDK_SCALE=2 

PROJECT=kicad/orb-50mm-3332led/

#FREEROUTING="java -jar freerouting_cli/freerouting_cli-1/build/obj/freerouting_cli.jar"
#FREEROUTING="java -jar freerouting-1.9.0/build/libs/freerouting-executable.jar"
FREEROUTING="java -jar freerouting-1.9.0/build/libs/freerouting-executable.jar"

${FREEROUTING} -de ${PROJECT}orb_generated_${1}_pass2.dsn -mp 5 -mt 0 -da -do ${PROJECT}orb_generated_${1}_power.ses
