#!/bin/bash

################# Change these #############################################################

# YCB
#results=results/pkpnet_ycbv_real+synt/pkpnet-epoch=59-nviews=-1-det=saved_ycbv-test
#scenes=( 48 49 50 51 52 53 54 55 56 57 58 59 )

# TLESS
results=results/pkpnet_tless_primesense+pbr/pkpnet-epoch=88-nviews=-1-det=saved-GT-CAM-POSE_tless-test_primesense+pbr
scenes=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
####################################################################################

outdir=$PWD/$results/scene_videos
mkdir -p $outdir

tmpfile=/tmp/vidlist.txt
if [ -f $tmpfile ]; then
    rm $tmpfile
fi

for scene in ${scenes[@]}; do
    infiles=$results/viz_images/scene_${scene}_%6d.png
    out=$outdir/scene_${scene}.mp4
    echo Creating single-scene video $out
    ffmpeg -y -i $infiles -c:v libx264 -vf "fps=10,format=yuv420p" $out #&> /dev/null
    echo file \'$out\' >> $tmpfile
done

# Combine all scenes into one video
out=$outdir/../full_video.mp4
ffmpeg -y -f concat -safe 0 -i $tmpfile -c copy $out
echo
echo
echo "=============================================================================="
echo Full video written to $out

xdg-open $out
