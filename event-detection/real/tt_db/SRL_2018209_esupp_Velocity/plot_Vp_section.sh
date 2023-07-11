#!/bin/sh
gmtset ANOT_FONT 1
gmtset ANOT_FONT_SIZE_PRIMARY 10
gmtset ANOT_OFFSET 3p

gmtset LABEL_FONT 1
gmtset LABEL_FONT_SIZE 10
gmtset LABEL_OFFSET 1p

gmtset HEADER_FONT 1
gmtset HEADER_FONT_SIZE 8
gmtset HEADER_OFFSET -12p

gmtset BASEMAP_TYPE plain
gmtset PAPER_MEDIA=A4
# gmtset TICK_LENGTH = 0.001i 
#
c0="-C93/28.5"   ##### longitude and latitude for the start point of profile
e0="-E133/47"  ##### longitude and latitude for the ending point of profile
c="93/133"     ##### longitudes for the start and end points of profileEseis
#
PROJ="-JX6.0i/-1.2i"
BOUNDS="-R$c/0/150" ##### plot depth range from 0  to 150 km        here 150 should be equal to the lowest depth number in the  line 37 
#
FNAME="vp_profile"
PSFILE=$FNAME.ps
PDFFILE=$FNAME.pdf
#
 # clean files
 ####  along  longitude 
if [ -e track.profile ]; then
    rm track.profile
fi
#  To generate points every 10 km along a great circle from 28.5N,93W to 47N,133W:   if we want to plot profile along the longitude or latitude, we should add -N in the project
   project  $c0 $e0 -G10 -Q > track.dat
  # iterate in depth
 for i in 0 5 10 15 20 30 40 60 80 100 120 150;do 
    f='Z_vp'$i''  
    gawk '{print $1,$2, $3}' $f > $f.txt
    xyz2grd $f.txt -R74.0/135.0/18.0/53.0 -I0.5/0.5 -G$f.nc
    grdtrack track.dat -G$f.nc | gawk -v dep=$i '{print $1, dep, $4}' >> track.profile
    rm  $f.nc
    rm  $f.txt
   done;
#######################   plot   profile velocity     -A0.25+s4p+gwhite 
f='track.profile'
surface $f  -T0.75 -G$f.grd  $BOUNDS -V -I0.02/0.1
grdsample $f.grd -Ginterpolated.grd -I0.02/0.01 -Ql
##### make cp.
makecpt -Cseis -T4.5/8.5/0.2 -D150/0/0/0/0/150 -Z  > vp.cpt
##### plot baseman
psbasemap $PROJ $BOUNDS -Ba5f1:"Longitude":/a20f10:"Depth(km)":nWSe:."":  -K  >$PSFILE 
####
grdimage $PROJ $BOUNDS  interpolated.grd -Cvp.cpt -V   -K -O >>$PSFILE 
####
grdcontour interpolated.grd  $PROJ $BOUNDS  -Gd5c -A0.5+s6p+gwhite -O -K  -W0.5p/255  >> $PSFILE
#####
psscale -Cvp.cpt -Ba0.4g0.2/:"Vp (km/s)": -D6.4i/0.6i/1.2i/0.08i  -O  >> $PSFILE
ps2raster -Tf -P $PSFILE
rm *.grd
