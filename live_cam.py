#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : Jun 01 12:40:32 2018
# Author   : Abhishek Kumar Bojja
# File     : live_cam.py

# Description:
# Maintainer:
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)

# Code:
#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : May 01 14:10:56 2018
# Author   : Abhishek Kumar Bojja
# File     : bad_image_remover.py

# Description:
# Maintainer:
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)

# Code:
import os
import sys
import cv2
import pyrealsense as pyrs
import torch
from models import Net
import numpy as np
from torch.autograd import Variable


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1
    else:
        print(directory, '- already exists')
        return 0


def check_directories():
    try:
        create_directory('./data')
        create_directory('./data/depth')
        create_directory('./data/color')
        create_directory('./data/cad')
        create_directory('./data/dac')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return -1
    return 0


def main():
    # file_structure = check_directories()
    # if file_structure == -1:
    #     print('\nERROR: Directories can\'t be created, error thrown')
    #     return -1
    # else:
    #     print('\nDirectories created successfully...\nLaunching camera module...')
    net = Net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # dev = torch.cuda.is_available()
    # # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #
    # print(dev)

    net.to(device)
    net.load_state_dict(torch.load('saved_models/keypoints_model_2.pt'))

    ## print out your net and prepare it for testing (uncomment the line below)
    net.eval()

    # net.load_state_dict(torch.load('saved_models/keypoints_model_2.pt'))

    # Fire camera & launch streams
    # pyrs.start()
    serv = pyrs.Service()
    # cam = pyrs.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps=60),
    #                                             pyrs.stream.DepthStream(fps=60),
    #                                             pyrs.stream.CADStream(fps=60),
    #                                             pyrs.stream.DACStream(fps=60)])
    cam = serv.Device(device_id=0, streams=[pyrs.stream.ColorStream(fps=60),
                                            # pyrs.stream.DepthStream(fps=60),
                                            # pyrs.stream.CADStream(fps=60),
                                            # pyrs.stream.DACStream(fps=60)
                                            ])
    # scale = cam.depth_scale

    # Some important variables
    flag_save_frames = False  #
    file_num = 0
    # cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    # out = cv2.VideoWriter('./output.avi', -1, 20.0, (640, 480))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('output_4.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (640, 480))

    # Start fetching Buffer
    print('Starting Buffer...')
    i = 1000
    while (i):
        cam.wait_for_frames()
        image_1 = cam.color[:, :, ::-1]
        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
        faces_1 = face_cascade.detectMultiScale(gray_1, 1.1, 5)

        # make a copy of the original image to plot detections on
        image_with_detections_1 = image_1.copy()

        # loop over the detected faces, mark the image where each face is found
        for (x, y, w, h) in faces_1:
            # face = gray_1
            roi = gray_1[y:y + int(h), x:x + int(w)]
            org_shape = roi.shape
            roi = roi / 255.0

            roi = cv2.resize(roi, (224, 224))
            # image_plot = np.copy(roi)
            roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
            roi = np.transpose(roi, (2, 0, 1))
            roi = torch.from_numpy(roi)
            roi = Variable(roi)
            roi = roi.type(torch.cuda.FloatTensor)
            roi = roi.unsqueeze(0)
            predicted_key_pts = net(roi)
            predicted_key_pts = predicted_key_pts.view(68, -1)
            predicted_key_pts = predicted_key_pts.data
            predicted_key_pts = predicted_key_pts.cpu().numpy()
            predicted_key_pts = predicted_key_pts * 50.0 + 100

            predicted_key_pts[:, 0] = predicted_key_pts[:, 0] * org_shape[0] / 224 + x
            predicted_key_pts[:, 1] = predicted_key_pts[:, 1] * org_shape[1] / 224 + y

            # cv2.rectangle(image_with_detections_1, (x, y), (x + w, y + h), (0, 0, 255), 3)

            for (x_point, y_point) in zip(predicted_key_pts[:, 0], predicted_key_pts[:, 1]):
                cv2.circle(image_with_detections_1, (x_point, y_point), 3, (0, 255, 0), -1)

        # current_color = cam.color[:, :, ::-1]
        # current_depth = cam.depth * scale
        # current_cad = cam.cad[:, :, ::-1]
        # current_dac = cam.dac * scale
        out.write(image_with_detections_1)
        cv2.imshow('Color', image_with_detections_1)
        # cv2.imshow('Depth', current_depth)
        # cv2.imshow('CAD', current_cad)
        # cv2.imshow('DAC', current_dac)

        # if flag_save_frames:
        #     num = format(file_num, '08')
        #     cv2.imwrite('./data/depth/' + str(num) + '.png', cam.depth)
        #     cv2.imwrite('./data/color/' + str(num) + '.png', current_color)
        #     cv2.imwrite('./data/dac/' + str(num) + '.png', cam.dac)
        #     cv2.imwrite('./data/cad/' + str(num) + '.png', current_cad)
        #     file_num += 1
        i = i-1
        k = cv2.waitKey(1)
        if k == ord('q'):
            print('Q Pressed...\nEnding execution')
            break
        if k == ord('f'):
            if flag_save_frames:
                print('F Pressed...\nStopped fetching frames...')
                flag_save_frames = False
            else:
                print('F Pressed...\nStarted fetching frames...')
                flag_save_frames = True

    cam.stop()
    # pyrs.stop()
    out.release()
    serv.stop()
    return 0


if __name__ == '__main__':
    print(__doc__)

    main()
