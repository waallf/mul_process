import glob as gb
import cv2
import os
import multiprocessing
def v2i(path,save_path,i,stride):
    print('start split {}'.format(i))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open('./douyin_list.txt','a') as f:
        vc = cv2.VideoCapture(path)
        c = 1
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        while rval :
            rval, frame = vc.read()
            c = c + 1
            if c%stride==0:
                f.write(save_path+"/"+str(c) + '.png'+'\n')
                cv2.imwrite(save_path +"/"+ str(c) + '.png', frame)

    print('end split '.format(i))

def i2v(path):
    videoWriter = cv2.VideoWriter('./3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (720,1280))

    for i in range(1,1000):
        img_path = path+'/{}_gt_res.png'.format(i)
        # img_path = i
        try:
            img  = cv2.imread(img_path)
            img = cv2.resize(img,(720,1280))
            videoWriter.write(img)
        except:
            print(i)
            print('sssssssss')
            continue
    print('done')
# bast_path = 'D:/data/suoer_image/round1/train\label'

if __name__=='__main__':
    base_path = './douyin_download_data/'
    save_path = './douyin_image/'
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    for path in os.listdir(base_path):
        i = int(path.split('.')[0])
        # v2i(base_path+path,save_path+path.split('.')[0],i,5)
        pool.apply_async(v2i,args=(base_path+path,save_path+path.split('.')[0],i,32))
    pool.close()
    pool.join()
# i2v('D:/edge-connect-master/result/3_sub1')