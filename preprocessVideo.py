import cv2
import moviepy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
import dlib

def get_alignments(align_file):
    f = open(align_file, 'r')
    lines = f.readlines()[1:]
    time_pairs = []
    for line in lines:
        if line.split()[0] != "Total":
            if line.split()[3] != '<s>' and line.split()[3] != '</s>' and line.split()[3] != '<sil>':
                print(line.split()[3]
                      )
                tp0 = ((int(line.split()[0]) + 2) / 100)
                tp1 = ((int(line.split()[1]) + 2) / 100)
                time_pairs.append([tp0, tp1])
        else:
            break
        return time_pairs


def process(text_file, video_file,align_file,label_file):
    with open(text_file, 'r') as f:
        content = f.read()

    #todo decipher the gender of the speaker using the text file. should i automate it or make it all by hand????? big question
    time_pairs=get_alignments(align_file)
    lines = content.splitlines()
    times=[]
    for line in lines:
        x = (line.split(': ')[0].strip().split(' ')[1])
        # print(x)
        # print(float(x.split('-')[0].split('[')[1]))
        times.append([float(x.split('-')[0].split('[')[1]), float(x.split('-')[1].split(']')[0])])
    main_features=[]
    with VideoFileClip(video_file) as video:
        for i in times:
            feats=[]
            new=video.subclip(i[0],i[1])
            new.write_videofile('video.mp4',audio_codec='aac')
            with VideoFileClip('video.mp4') as sentence_video:
                for k in time_pairs:
                    word_video=sentence_video.subclip(time_pairs[0],time_pairs[1])
                    word_video.write('word_video.mp4',audio_codec='aac')
                    fImages = []
                    mImages = []
                    cap = cv2.VideoCapture('testing.mp4')
                    hog_face_detector = dlib.get_frontal_face_detector()
                    while True:
                        ret, img = cap.read()
                        # print(img)
                        if isinstance(img, type(None)):

                            break
                        else:

                            height, width = img.shape[:2]

                            start_row, start_col = int(0), int(0)

                            end_row, end_col = int(height), int(width * 0.5)
                            cropped_top = img[start_row:end_row, start_col:end_col]

                            faces_hog = hog_face_detector(cropped_top, 1)

                            for face in faces_hog:
                                x = face.left()

                                y = face.top()

                                w = face.right() - x

                                h = face.bottom() - y

                                cropping = cropped_top[y:y + h, x:x + w]

                                cropping = cv2.resize(cropping, (125, 125))
                                # cv2_imshow(cropping)
                                fImages.append(cropping)
                            # cv2_imshow(cropped_top)

                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                            # Let's get the starting pixel coordiantes (top left of cropped bottom)
                            start_row, start_col = int(0), int(width * 0.5)
                            # Let's get the ending pixel coordinates (bottom right of cropped bottom)
                            end_row, end_col = int(height), int(width)
                            cropped_bot = image[start_row:end_row, start_col:end_col]
                            # print(start_row, end_row )
                            # print(start_col, end_col)
                            mImages.append(cropped_bot)
                            # cv2_imshow( cropped_bot)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()



