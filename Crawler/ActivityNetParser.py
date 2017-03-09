import json
from pprint import pprint
import glob
import os
import shutil
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess

with open('activity_net.v1-3.min.json') as data_file:    
    data = json.load(data_file)

if os.path.isdir('training')==0:
	os.mkdir('training')
if os.path.isdir('testing')==0:
	os.mkdir('testing')
if os.path.isdir('validation')==0:
	os.mkdir('validation')

dataref=data["database"]
pprint(len(dataref))
videos= glob.glob("*.mp4")
str1="t e s t i n g"
str2="t r a i n i n g"
str3="v a l i d a t i o n"
print('The number of videos are',len(videos))
for i in range(0,len(videos)):

	print('Processing video number ',i)
	video=videos[i]
	vl1=dataref[video[2:len(video)-4]]["annotations"]
	v2=dataref[video[2:len(video)-4]]["annotations"]
	seglen=len(v2)	
	print('The video is ', video)	
	subs=dataref[video[2:len(video)-4]]["subset"]
	vl= " ".join(str(x) for x in vl1)
	s1= " ".join(str(x) for x in subs)
	
	if s1==str1:
		print('testing')
		shutil.copy2(video,'testing')

	if s1==str2:
		print('training')
		start = vl.find('label')+10
        end=vl.find('}',start)-1
        label=vl[start:end]
		newdir='training'+'/'+label
		if os.path.isdir(newdir)==0:
			os.mkdir(newdir)
		for j in range(0,seglen):
            stri=v2[j].items()
            seg=stri[0]
            t=seg[1]
            tstart=t[0]
            tend=t[1]
			vstr=newdir+'/'+video[0:len(video)-4]+str(j)+'.mp4'
			ffmpeg_extract_subclip(video, tstart, tend, targetname=vstr)
	if s1==str3:
		print('validation')
		start = vl.find('label')+10
		end=vl.find('}',start)-1
		label=vl[start:end]
		newdir='validation'+'/'+label
		if os.path.isdir(newdir)==0:
			os.mkdir(newdir)
		for j in range(0,seglen):
            stri=v2[j].items()
            seg=stri[0]
            t=seg[1]
            tstart=t[0]
            tend=t[1]
			vstr=newdir+'/'+video[0:len(video)-4]+str(j)+'.mp4'
			ffmpeg_extract_subclip(video, tstart, tend, targetname=vstr)
	

