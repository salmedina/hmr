"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()


def visualize_all(num_persons, img, proc_params, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    skel_img = np.copy(img)
    rend_img_overlay = np.copy(img)
    rend_img = np.zeros(shape=img.shape)
    # rend_img_vp1 = np.zeros(shape=img.shape)
    # rend_img_vp2 = np.zeros(shape=img.shape)

    for idx in range(num_persons):
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_params[idx], verts[idx], cam[idx], joints[idx], img_size=img.shape[:2])

        # Render results
        skel_img = vis_util.draw_skeleton(skel_img, joints_orig)
        rend_img_overlay = renderer(
            vert_shifted, cam=cam_for_render, img=rend_img_overlay, do_alpha=True, color_id=idx)
        rend_img_overlay = rend_img_overlay[:, :, :3]
        rend_img = renderer(
            vert_shifted, cam=cam_for_render, img=rend_img, img_size=img.shape[:2], color_id=idx)
        # rend_img_vp1 = renderer.rotated(
        #     vert_shifted, 60, cam=cam_for_render, img=rend_img_vp1, img_size=img.shape[:2])
        # rend_img_vp2 = renderer.rotated(
        #     vert_shifted, -60, cam=cam_for_render, img=rend_img_vp2, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(skel_img)
    plt.title('Joints  2D Projection')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh Overlay')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(rend_img)
    plt.title('3D Mesh')
    plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(rend_img_vp1)
    # plt.title('diff vp')
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(rend_img_vp2)
    # plt.title('diff vp')
    # plt.axis('off')
    plt.draw()
    plt.show()


def preprocess_image(img_path, person_bbox=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if person_bbox is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        x1, y1, x2, y2 = person_bbox
        center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
        person_height = np.linalg.norm(y2 - y1)
        scale = 150. / person_height

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path):
    sess = tf.compat.v1.Session()

    # meva_sample_1: person_bboxes = [[171, 102, 225, 244], [63, 71, 104, 199]]
    # meva sample 2: person_bboxes = [[95, 132, 429, 551], [0, 2, 245, 485], [319, 43, 539, 427]]
    # meva_sample 3: person_bboxes = [[155, 224, 381, 508], [19, 112, 238, 499], [305, 158, 508, 404]]

    person_bboxes = [[319, 43, 539, 427], [0, 2, 245, 485], [95, 132, 429, 551]]
    num_persons = len(person_bboxes)

    # Demo only processes one image at a time
    config.batch_size = num_persons
    model = RunModel(config, sess=sess)

    input_array = np.zeros(shape=[num_persons, config.img_size, config.img_size, 3])
    proc_params = []
    for person_idx, person_bbox in enumerate(person_bboxes):
        input_img, proc_param, img = preprocess_image(img_path, person_bbox)
        proc_params.append(proc_param)
        # Add batch dimension: 1 x D x D x 3
        input_array[person_idx] = input_img
        #input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_array, get_theta=True)

    visualize_all(num_persons, img, proc_params, joints, verts, cams)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)

    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    # Global renderer needs to be declared
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path)
