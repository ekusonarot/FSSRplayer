#videos="HH/HH_270p.mp4 HH/HH_180p.mp4 HH/HH_144p.mp4 \
#ToS/ToS_270p.mp4 ToS/ToS_180p.mp4 ToS/ToS_144p.mp4 \
#BBB30fps/BBB30fps_144p.mp4 \
videos="BBB60fps/BBB60fps_144p.mp4"
#BBB24fps/BBB24fps_144p.mp4 BBB24fps/BBB24fps_180p.mp4 BBB24fps/BBB24fps_270p.mp4"
for video in $videos
do
name=(${video//// })
#python3 FSSRplayer.py -m FSSRv2 -c -u http://192.168.1.142/$video -e -s 10
#python3 eval.py -i videolog/${name[1]}/FSSRv2_10ign10buftime.avi -gt GT/GT_${name[1]}
#rm videolog/${name[1]}/FSSRv2_10ign10buftime.avi
#python3 FSSRplayer.py -m NSSRv2 -c -u http://192.168.1.142/$video -e -s 10
#python3 eval.py -i videolog/${name[1]}/NSSRv2_10ign10buftime.avi -gt GT/GT_${name[1]}
#rm videolog/${name[1]}/NSSRv2_10ign10buftime.avi
python3 FSSRplayer.py -m BIC -c -u http://192.168.1.142/$video
python3 eval.py -i videolog/${name[1]}/BIC.avi -gt GT/GT_${name[1]}
rm videolog/${name[1]}/BIC.avi
done
<<COMMENT
segment_sizes="5 10 15 20 25 30 35 40 45 50"
for size in $segment_sizes
do
python3 FSSRplayer.py -m FSSRv2 -c -u http://192.168.1.142/BBB24fps/BBB24fps_144p.mp4 -e -s $size
python3 eval.py -i videolog/BBB24fps_144p.mp4/FSSRv2_10ign${size}buftime.avi -gt GT/GT_BBB24fps_144p.mp4
rm videolog/BBB24fps_144p.mp4/FSSRv2_10ign${size}buftime.avi
done
COMMENT
<< COMMENT
python3 FSSRplayer.py -m FSSRv2 -c -u http://192.168.1.142/ToS/ToS_270p.mp4 -e
python3 eval.py -i videolog/ToS_270p.mp4/FSSRv2_10ign20buftime.avi -gt GT/GT_ToS_270p.mp4
rm videolog/ToS_270p.mp4/FSSRv2_10ign20buftime.avi
python3 FSSRplayer.py -m NSSRv2 -c -u http://192.168.1.142/ToS/ToS_270p.mp4 -e
python3 eval.py -i videolog/ToS_270p.mp4/NSSRv2_10ign20buftime.avi -gt GT/GT_ToS_270p.mp4
rm videolog/ToS_270p.mp4/NSSRv2_10ign20buftime.avi

python3 FSSRplayer.py -m FSSRv2 -c -u http://192.168.1.142/BBB24fps/BBB24fps_270p.mp4 -e
python3 eval.py -i videolog/BBB24fps_270p.mp4/FSSRv2_10ign20buftime.avi -gt GT/GT_BBB24fps_270p.mp4
rm videolog/BBB24fps_270p.mp4/FSSRv2_10ign20buftime.avi
python3 FSSRplayer.py -m NSSRv2 -c -u http://192.168.1.142/BBB24fps/BBB24fps_270p.mp4 -e
python3 eval.py -i videolog/BBB24fps_270p.mp4/NSSRv2_10ign20buftime.avi -gt GT/GT_BBB24fps_270p.mp4
rm videolog/BBB24fps_270p.mp4/NSSRv2_10ign20buftime.avi

python3 FSSRplayer.py -m NSSRv2 -c -u http://192.168.1.142/HH/HH_270p.mp4 -e
python3 eval.py -i videolog/HH_270p.mp4/NSSRv2_10ign20buftime.avi -gt GT/GT_HH_270p.mp4
rm videolog/HH_270p.mp4/NSSRv2_10ign20buftime.avi
COMMENT