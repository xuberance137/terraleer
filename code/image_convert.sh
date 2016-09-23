#!/bin/bash
#selecting bands and reprojecting
for BAND in {4,3,2}; do
  gdalwarp -t_srs EPSG:3857 ../data/sceneData/LC80250302015139LGN00/LC80250302015139LGN00_B$BAND.TIF $BAND-projected.tif;
done
#combining to RGB image
convert -combine {4,3,2}-projected.tif RGB.tif
#we need to increase both brightness and contrast to compensate for bright spots on earth like Greenland
#account for haze by lowering the blue channel’s gamma (brightness) slightly, and raising the red channel’s even less, before increasing the contrast
convert -channel B -gamma 0.925 -channel R -gamma 1.03 -channel RGB -sigmoidal-contrast 50x16% RGB.tif RGB-corrected.tif
#convertingfrom 16 bit to 8 bit image
convert -depth 8 RGB-corrected.tif RGB-corrected-8bit.tif