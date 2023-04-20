import numpy as np
import cv2
import yaml




def read_intrinsic(file_path):
    intrinsic  = np.zeros((3, 3))
    distortion = np.zeros((5, 1))
    with open(file_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        for j in range(3):
            intrinsic[j, 0:3] = [   result['camera_matrix']['data'][j*3],
                                    result['camera_matrix']['data'][j*3+1],
                                    result['camera_matrix']['data'][j*3+2]
                                ]
        for j in range(5):
            distortion[j, 0] = result['distortion_coefficients']['data'][j]
    return intrinsic, distortion




def main_rgb():
    extrinsic = np.asarray([
                            [  1.0000,   -0.0050,    0.0047,    0.5042],
                            [  0.0050,    0.9999,    0.0113,    0.0031],
                            [ -0.0048,   -0.0113,    0.9999,   -0.0069],
                            [       0,         0,         0,    1.0000]
                            ])

    extrinsic = np.linalg.inv(extrinsic)

    rgb_image_left_path  = "1630106835306123587_left.png"
    rgb_image_right_path = "1630106835306123587_right.png"
    extrinsic_path_rgb   = "calibration/extrinsic/rgb_right_2_rgb_left"

    left_intrinsic_path  = "calibration/left.yaml"
    right_intrinsic_path = "calibration/right.yaml"

    left_intrinsic,  left_distortion  = read_intrinsic(left_intrinsic_path)
    right_intrinsic, right_distortion = read_intrinsic(right_intrinsic_path)

    rgb_image_left  = cv2.imread(rgb_image_left_path)
    rgb_image_right = cv2.imread(rgb_image_right_path)



    image_size = (1280, 560)

    rotation_matrix    = extrinsic[0:3, 0:3]
    translation_vector = extrinsic[0:3, 3]



    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(  left_intrinsic, left_distortion,
                                                        right_intrinsic, right_distortion,
                                                        image_size, rotation_matrix, translation_vector,
                                                        flags=cv2.CALIB_ZERO_DISPARITY,
                                                        newImageSize=image_size)



    left_map1,  left_map2  = cv2.initUndistortRectifyMap(left_intrinsic,  left_distortion,  R1, P1, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_intrinsic, right_distortion, R2, P2, image_size, cv2.CV_16SC2)

    # left_rectified  = cv2.remap(thermal_image_left_uint8, left_map1, left_map2, cv2.INTER_LANCZOS4)
    # right_rectified = cv2.remap(thermal_image_right_uint8, right_map1, right_map2, cv2.INTER_LANCZOS4)

    left_rectified  = cv2.remap(rgb_image_left, left_map1, left_map2, cv2.INTER_LANCZOS4)
    right_rectified = cv2.remap(rgb_image_right, right_map1, right_map2, cv2.INTER_LANCZOS4)

    canvas = np.concatenate([left_rectified, right_rectified], axis=1)

    cv2.line(canvas, (0, 100), (canvas.shape[1], 100), (255, 0, 0), 1)
    cv2.line(canvas, (0, 200), (canvas.shape[1], 200), (255, 0, 0), 1)
    cv2.line(canvas, (0, 300), (canvas.shape[1], 300), (255, 0, 0), 1)
    cv2.line(canvas, (0, 400), (canvas.shape[1], 400), (255, 0, 0), 1)

    # Display the rectified images (optional)
    cv2.imshow('rectified_uint8', canvas)

    cv2.imwrite("left_rectified.png",  left_rectified)
    cv2.imwrite("right_rectified.png", right_rectified)


    DO_SGBM = 1

    if DO_SGBM:
        max_disparity   = 128
        stereoProcessor = cv2.StereoSGBM_create(minDisparity = 0,
                                                numDisparities = max_disparity,
                                                blockSize = 9, 
                                                # uniquenessRatio = 1,
                                                )
        disparity = stereoProcessor.compute(left_rectified, right_rectified)

        cv2.filterSpeckles(img = disparity,
                            newVal = 0,
                            maxSpeckleSize = 60,
                            maxDiff = max_disparity,
                            )

        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)

        disparity_scaled = (disparity / 16.).astype(np.uint8)
        disparity_colour_mapped = cv2.applyColorMap(
            (disparity_scaled * (256. / max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT)
        cv2.imshow("colored_disparity", disparity_colour_mapped)
        cv2.imwrite("disparity_colour_mapped.png", disparity_colour_mapped)


    cv2.waitKey(0)






def main_thermal():
    extrinsic = np.asarray([
                            [    0.9998,   -0.0036,    0.0210,    0.6150],
                            [    0.0036,    1.0000,   -0.0010,   -0.0033],
                            [   -0.0210,    0.0011,    0.9998,   -0.0339],
                            [         0,         0,         0,    1.0000]
                        ])
    extrinsic = np.linalg.inv(extrinsic)
    thermal_image_left_path  =  "1630106835318041837.png"
    thermal_image_right_path = "1630106835327677483.png"
    extrinsic_path_thermal   = "calibration/extrinsic/thermal_right_2_thermal_left"

    left_intrinsic_path  = "calibration/thermal_14bit_left.yaml"
    right_intrinsic_path = "calibration/thermal_14bit_right.yaml"

    left_intrinsic,  left_distortion  = read_intrinsic(left_intrinsic_path)
    right_intrinsic, right_distortion = read_intrinsic(right_intrinsic_path)

    thermal_image_left  = cv2.imread(thermal_image_left_path, -1)
    thermal_image_right = cv2.imread(thermal_image_right_path, -1)

    thermal_image_left_uint8  = cv2.normalize(thermal_image_left, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    thermal_image_right_uint8 = cv2.normalize(thermal_image_right, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    cv2.imshow("unrectified_uint8", np.concatenate([thermal_image_left_uint8, thermal_image_right_uint8], axis=1))
    
    image_size = (640, 512)

    rotation_matrix    = extrinsic[0:3, 0:3]
    translation_vector = extrinsic[0:3, 3]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(  left_intrinsic, left_distortion,
                                                        right_intrinsic, right_distortion,
                                                        image_size, rotation_matrix, translation_vector,
                                                        flags=cv2.CALIB_ZERO_DISPARITY,
                                                        newImageSize=image_size)


    left_map1,  left_map2  = cv2.initUndistortRectifyMap(left_intrinsic,  left_distortion,  R1, P1, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_intrinsic, right_distortion, R2, P2, image_size, cv2.CV_16SC2)

    left_rectified  = cv2.remap(thermal_image_left_uint8, left_map1, left_map2, cv2.INTER_LANCZOS4)
    right_rectified = cv2.remap(thermal_image_right_uint8, right_map1, right_map2, cv2.INTER_LANCZOS4)

    canvas = np.concatenate([left_rectified, right_rectified], axis=1)

    cv2.line(canvas, (0, 100), (canvas.shape[1], 100), (255, 0, 0), 1)
    cv2.line(canvas, (0, 200), (canvas.shape[1], 200), (255, 0, 0), 1)
    cv2.line(canvas, (0, 300), (canvas.shape[1], 300), (255, 0, 0), 1)
    cv2.line(canvas, (0, 400), (canvas.shape[1], 400), (255, 0, 0), 1)

    # Display the rectified images (optional)
    cv2.imshow('rectified_uint8', canvas)

    cv2.imwrite("left_rectified.png",  left_rectified)
    cv2.imwrite("right_rectified.png", right_rectified)


    DO_SGBM = 1

    if DO_SGBM:
        max_disparity   = 128
        stereoProcessor = cv2.StereoSGBM_create(minDisparity = 0,
                                                numDisparities = max_disparity,
                                                blockSize = 9, 
                                                # uniquenessRatio = 1,
                                                )
        disparity = stereoProcessor.compute(left_rectified, right_rectified)

        cv2.filterSpeckles(img = disparity,
                            newVal = 0,
                            maxSpeckleSize = 60,
                            maxDiff = max_disparity,
                            )

        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)

        disparity_scaled = (disparity / 16.).astype(np.uint8)
        disparity_colour_mapped = cv2.applyColorMap(
            (disparity_scaled * (256. / max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT)
        cv2.imshow("colored_disparity", disparity_colour_mapped)
        cv2.imwrite("disparity_colour_mapped.png", disparity_colour_mapped)

    cv2.waitKey(0)


if __name__ == '__main__':




    main_thermal()



