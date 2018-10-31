import argparse
import cv2
import hickle as hkl
import numpy as np
import os

from demo_superpoint import SuperPointFrontend, VideoStreamer, myjet

if __name__ == '__main__':

  # Parse command line arguments.
  # Same as in demo:
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('input', type=str, default='',
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  # Different from demo:
  parser.add_argument('--write_dir', type=str, default='parse_outputs/',
      help='Directory where to write output frames (default: tracker_outputs/).')
  opt = parser.parse_args()
  print(opt)

  # This class helps load input images from different sources.
  vs = VideoStreamer(opt.input, opt.camid, None, None, opt.skip, opt.img_glob)
  assert not vs.video_file

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4

  # Create output directory.
  print('==> Will write outputs to %s' % opt.write_dir)
  if not os.path.exists(opt.write_dir):
    os.makedirs(opt.write_dir)

  print('==> Parsing %s.' % opt.input)
  while True:
    # Get a new image.
    img, status = vs.next_frame()
    if status is False:
      break

    pts, desc, heatmap = fe.run(img)

    # Primary output - Show point tracks overlayed on top of input image.
    out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    # Extra output -- Show current point detections.
    out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    for pt in pts.T:
      pt1 = (int(round(pt[0])), int(round(pt[1])))
      cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
    cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show the point confidence heatmap.
    if heatmap is not None:
      min_conf = 0.001
      heatmap[heatmap < min_conf] = min_conf
      heatmap = -np.log(heatmap)
      heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
      out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
      out3 = (out3*255).astype('uint8')
      out3 = cv2.resize(out3, (out2.shape[1], out2.shape[0]))
    else:
      out3 = np.zeros_like(out2)
    cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

    # Resize final output.
    try:
      out = np.hstack((out1, out2, out3))
    except ValueError:
      print('Heatmap dim is %d, %d' % (heatmap.shape[0], heatmap.shape[1]))
      raise Exception('Cannot hstack %d,%d %d,%d, %d,%d' % \
        (out1.shape[0], out1.shape[1], 
        out2.shape[0], out2.shape[1], 
        out3.shape[0], out3.shape[1]))
    
    # Write images to disk.
    out_root = os.path.join(opt.write_dir, os.path.split(vs.listing[vs.i-1])[1])
    out_file = out_root + '.png'
    print('Writing image to %s' % out_file)
    cv2.imwrite(out_file, out)
    
    # Write hickle.
    hkl_out = out_root + '.hkl'
    hkl.dump([heatmap, np.vstack((pts, desc)).T], open(hkl_out, 'w'))

  print('==> Finshed Parsing.')
