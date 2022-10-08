#!/usr/bin/env python3

# Copyright 2020 nakaterketo

"""
=====
Title
=====

Description

"""

from argparse import ArgumentParser

import cv2
import numpy as np
import sys
import time


class FramesPerSecond(object):
    def __init__(self, frames_number):
        self.frames_number = frames_number
        self.sum_time = 0
        self.counter = 0
        self.value = 0
        self.prev_time = None

    def set_time(self):
        current_time = time.time()
        self.counter += 1
        if self.prev_time is not None:
            self.sum_time = self.sum_time + current_time - self.prev_time
            if self.counter % self.frames_number == 0:
                self.value = int(self.frames_number / self.sum_time)
                self.sum_time = 0
        self.prev_time = current_time


def put_png(frame, coords, png):
    for coord in coords:
        png = cv2.resize(png, (coord[2], coord[3]), interpolation=cv2.INTER_LINEAR)
        ind = np.where(png[:,:,3] == 255)
        frame[ind[0]+coord[1], ind[1]+coord[0]] = png[:,:,:3][ind]
    return frame


def main():
    description = 'Tool, detecting moving object'
    parser = ArgumentParser(description=description)

    parser.add_argument('--video', default=0,
                        help='Video file (camera if not specified)')

    parser.add_argument('--width', default=1280, type=int,
                        help='Width of video to be processed'
                        ' (default: %(default)s)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, init_frame = cap.read()
    if not ret:
        print('Error: file {} is empty'.format(args.video))
        sys.exit(1)
    aspect_ratio = init_frame.shape[1] / init_frame.shape[0]
    width = args.width
    height = int(width / aspect_ratio)
    init_frame = cv2.resize(init_frame, (width, height), interpolation=cv2.INTER_CUBIC)

    fps = FramesPerSecond(30)

    haar_face_cascade = cv2.CascadeClassifier('haar-eyes.xml')

    png = cv2.imread('eye2.png', cv2.IMREAD_UNCHANGED)

    while True:
        fps.set_time()

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        faces = haar_face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        frame = put_png(frame, faces, png)
        #for face in faces:
            #cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 3)

        cv2.putText(frame, 'Frame: {}'.format(fps.counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'FPS: {}'.format(fps.value), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
