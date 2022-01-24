import os
import cv2
from base_camera import BaseCamera
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from track import *
from yolov5.utils.datasets import *
from yolov5.utils.utils import *
import torch.backends.cudnn as cudnn

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, config_deepsort = \
        'inference/output', '0','yolov5/weights/yolov5m.pt','deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', 'store_false', 'store_true', \
                'store_true', 600, 'store_true',"deep_sort_pytorch/configs/deep_sort.yaml"
        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = torch_utils.select_device()

        # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
        # its own .txt file. Hence, in that case, the output folder is not restored
        if not evaluate:
            if os.path.exists(out):
                pass
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder
        half = device.type != 'cpu' # half precision only supported on CUDA

        # Load model
        model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Check if environment supports images displays
        if show_vid:
            show_vid = check_imshow()

        if webcam:
            cudnn.benchmark = True  # set True to speed up constant images size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        save_path = str(Path(out))
        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5,
                                fast=True, classes=None, agnostic=False)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per images
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    xywh_bboxs = []
                    confs = []
                    classes = []


                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        # to deep sort format
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        classes.append([cls.item()])

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    classes = torch.Tensor(classes)
                    # pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, classes, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, 4]
                        classes = outputs[:,5]

                        draw_boxes(im0, bbox_xyxy,[names[i] for i in classes], identities)

                        # to MOT format
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                        # Write MOT compliant results to file
                        if save_txt:
                            for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                                bbox_top = tlwh_bbox[0]
                                bbox_left = tlwh_bbox[1]
                                bbox_w = tlwh_bbox[2]
                                bbox_h = tlwh_bbox[3]
                                identity = output[-1]
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                                bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                # Save results (images with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

 
            print('Done. (%.3fs)' % (time.time() - t0))
            yield cv2.imencode('.jpg', im0)[1].tobytes()

    # 기본으로 지정'--yolo_weights' '--show-vid' '--device'
    # 사용 명령어     python track.py --source invade.mp4 --img 640
