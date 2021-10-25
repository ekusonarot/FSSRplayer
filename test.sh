urls="https://youtu.be/BBvod49uySQ"
for url in $urls
do
python3 FSSRplayer.py -m FSSR -c -u $url -e
done

videonames="Best\ 1\ minute\ INSPIRATIONAL\ video\ \|\ Priyanshu"
for url in $urls
do
python3 eval.py -i videolog/$videonames/FSSR_10ign20buftime.avi -gt GT/GT_$videonames.mp4
done