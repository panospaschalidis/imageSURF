import cv2
import os
import argparse
import numpy as np

from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type=str,default=None)
    parser.add_argument('--sub_sample',action="store_true",default=False)
    
    args = parser.parse_args()
    if not args.video_path:
        print('Please provide a valid video path')
        raise NotImplementedError

    cam = cv2.VideoCapture(args.video_path)
    im_dir = os.path.splitext(os.path.basename(args.video_path))[0]
    os.makedirs(im_dir, exist_ok=True)
    frameno = 0
    while(True):
       ret,frame = cam.read()
       
       
       if ret:
          # if video is still left continue creating images
          name = os.path.join(os.getcwd(),im_dir,'frame_' +str(frameno).zfill(6) + '.png')
          print ('new frame captured...' + name)
          if args.sub_sample:
            frame = np.array(Image.fromarray(frame).resize((int(frame.shape[1]/2), int(frame.shape[0]/2))))
            cv2.imwrite(name, frame)
          else:
            cv2.imwrite(name, frame)
          frameno += 1
       else:
          break

    cam.release()
    cv2.destroyAllWindows()

