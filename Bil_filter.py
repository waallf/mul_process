import math
from PIL import Image
import numpy as np
import cv2
class Gaosi():
    def __init__(self,gama,k):
        self.gama = gama
        self.k = k

    def gaosi(self,point,img,is_shu):
        W = self.allocate_w()
        x,y = point
        num=0
        if is_shu:# 模糊上下
            num = np.sum(img[x-self.k//2:x+self.k//2+1,y-self.k//2:y+self.k//2+1] *W,axis=(0,1))
        img[x][y] =num
        return img

    def allocate_w(self):
        weight = [[0 for  i in range(self.k)] for _ in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                weight[i][j]=self.kernal(i-self.k//2,j-self.k//2)
        weight = self.softmax(weight)
        weight = np.expand_dims(weight,-1)
        weight = np.tile(weight,(1,1,3))
        # weight = np.transpose(weight,(1,2,0))
        return weight
    def dan_allocate_w(self):
        weight = [0 for  i in range(self.k)]

        for i in range(self.k):
                weight[i]=self.dan_kernal(i-self.k//2)
        weight = self.softmax(weight)
        weight = np.expand_dims(weight,-1)
        return weight

    def dan_kernal(self,x):
        return math.e**(-(x**2)/2*self.gama**2)/self.gama*math.sqrt(2*math.pi)

    def kernal(self,x,y):
        return (math.e**((-(x**2+(y-2)**2))/2*self.gama**2))/self.gama*math.sqrt(2*math.pi)

    def softmax(self,x):
        """Compute the softmax of vector x."""
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def g_mask(self,img,start_point,m_,n_,num_clip):
        org_mask = np.ones(img.shape[:2])
        m,n = m_+40,n_+40
        mask = np.ones((n_+40,m_+40))
        # mask = np.ones((n,m))
        # W = self.dan_kernal(num_clip)
        W = [i/20 for i in range(20,0,-1)]
        res = []
        j=0
        for o in range((min(m, n) + 1) // 2):
            [res.append([o, i]) for i in range(o, m - o)]
            [res.append([j, m - 1 - o]) for j in range(o, n - o) if [j, m - 1 - o] not in res]
            [res.append([n - 1 - o, k]) for k in range(m - 1 - o, o - 1, -1) if [n - 1 - o, k] not in res]
            [res.append([l, o]) for l in range(n - 1 - o, o - 1, -1) if [l, o] not in res]
            if j>=len(W):
                for point in res:
                    mask[point[0]][point[1]] =0
                res = []
            else:
                for point in res:
                    try:
                        mask[point[0]][point[1]]*=W[j]
                    except:
                        print(point)
                        print(j)
                res = []
                j+=1
        mask = mask.T
        org_mask[start_point[0]-20:start_point[0]+m_+20,start_point[1]-20:start_point[1]+n_+20] = mask
        return org_mask

def get_points():
    points = []
    for j in range(20,80):
        for i in range(1510,1900):
            points.append((j,i))
    for j in range(70, 130):
        for i in range(1510, 1900):
            points.append((j, i))
    return points

class Junzhi():
    def __init__(self,k,img,mask_satrt,mask_h,mask_w,gama =1,):
        self.k = k
        self.k_w = k-2
        self.img = img
        self.mask_satrt=mask_satrt
        mask_center = (mask_satrt[0]+mask_h//2,mask_satrt[1]+mask_w//2)
        self.mask_center = mask_center
        self.mask_h = mask_h
        self.mask_w =mask_w
        # self.W = self.alloca_w()
        self.gama = 1
        self.W_up = np.array(self.allocate_w(1,0),dtype=np.float)
        self.W_left = np.array(self.allocate_w(0,1),dtype=np.float)
        self.gray_img  = np.array(cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY),dtype=np.float)

    def alloca_w(self):
        W = [1 for _ in range(self.k)]
        W[40]=0
        W = np.expand_dims(W,-1)
        return W
    #####两维高斯
    def allocate_w(self,u_d,l_r):
        weight = [[0 for i in range(self.k_w)] for _ in range(self.k )]
        for i in range(self.k ):
            for j in range(self.k_w ):
                weight[i][j] = self.kernal(i - self.k // 2, j - self.k_w  // 2,u_d,l_r)
        # weight = self.softmax(weight)
        # weight = np.expand_dims(weight, -1)
        # weight = np.tile(weight, (1, 1, 3))
        # weight = np.transpose(weight,(1,2,0))
        return weight

    def kernal(self, x, y,u_d,l_r):
        return (math.e ** ((-((x+u_d) ** 2 + (y+l_r) ** 2)) / 2 * (self.gama ** 2))) / self.gama * math.sqrt(2 * math.pi)

    #####一维高斯
    def dan_allocate_w(self):
        weight = [0 for  i in range(self.k)]
        for i in range(self.k):
                weight[i]=self.dan_kernal(self.k-i)
        weight = self.softmax(weight)
        weight = np.expand_dims(weight,-1)
        return weight

    def dan_kernal(self,x):
        return math.e**(-(x**2)/2*self.gama**2)/self.gama*math.sqrt(2*math.pi)

    def softmax(self,x):
        """Compute the softmax of vector x."""
        # exp_x = np.exp(x)
        # print(x)
        softmax_x = x / np.sum(x)
        return softmax_x

    # 灰度值权重
    def allocate_hw(self,c_x,c_y):
        weight = [[0 for _ in range(self.k_w)] for _ in range(self.k )]
        for i in range(self.k):
            for j in range(self.k_w):
                tmp = -(self.gray_img[c_x+i-self.k//2][c_y+j-self.k_w //2]-self.gray_img[c_x][c_y])**2
                weight[i][j] = math.e **(tmp/(2*self.gama**2))
        return weight

    # 双边滤波权重
    def bilater(self,c_x,c_y,alpha,direction):
        H_w = self.allocate_hw(c_x,c_y)
        if direction=='up':
            w = self.W_up
            weight = w * np.asarray(H_w)
            weight[:self.k // 2] = weight[:self.k // 2] * alpha

        elif direction=='down':
            w = self.W_up[::-1]
            weight = w * np.asarray(H_w)
            weight[self.k // 2:] = weight[self.k // 2:] * alpha

        if direction=='left':
            w = self.W_left
            weight = w * np.asarray(H_w)
            weight[:][:self.k // 2] = weight[:self.k // 2] * alpha

        elif direction=='right':
            w = self.W_left[::-1]
            weight = w * np.asarray(H_w)
            weight[:][self.k // 2:] = weight[self.k // 2:] * alpha
        else:
            assert ('Direction False')

        weight = w* np.asarray(H_w)
        # weight =w.copy()
        #####################################
        weight[:self.k // 2] = weight[:self.k // 2]*alpha
        w = self.softmax(weight)
        w = np.expand_dims(w, -1)
        w = np.tile(w, (1, 1, 3))
        return w

    def compute_alpha(self,point,direction):
        if direction == 'up' or direction == 'down':
            mask_d = self.mask_h//2
            d = abs(point[0]-self.mask_center[0])
        elif direction == 'left' or direction == 'right':
            mask_d = self.mask_w//2
            d = abs(point[1]-self.mask_center[1])
        else:
            assert ('Direction False')

        dif = d/mask_d
        alpha = 1.5* dif
        return alpha

    def get_points(self):
        points= []
        for i in range(self.mask_h//2+20):
            for j in range(self.mask_w):
                points.append(([self.mask_satrt[0] + i-20, self.mask_satrt[1] + j],'up'))

        for i in range(self.mask_h,self.mask_h//2-20,-1):
            for j in range(self.mask_w):
                points.append(([self.mask_satrt[0] + i+20, self.mask_satrt[1] + j],'down'))

        for i in range(self.mask_h):
            for j in range(self.mask_w//2+20):
                points.append(([self.mask_satrt[0] + i, self.mask_satrt[1] + j-20],'left'))

        for i in range(self.mask_h):
            for j in range(self.mask_w,self.mask_w//2-20,-1):
                points.append(([self.mask_satrt[0] + i, self.mask_satrt[1] +j + 20], 'right'))

        for i in range(self.mask_h // 2 - 7, self.mask_h // 2 + 7):
            for j in range(self.mask_w):
                points.append(([self.mask_satrt[0] + i, self.mask_satrt[1] + j], 'up'))
        return points

    def k_mean(self):
        points = self.get_points()
        for index,(point,direction) in enumerate(points):
            # print(index)
            x,y = point
            alpha = self.compute_alpha(point, direction)
            w = self.bilater(x, y, alpha, direction)
            if direction=='right':
                print()
            if direction=='down':
                num = np.sum(self.img[x - self.k//2+1:x+self.k//2+1+1, y - self.k_w//2:y+self.k_w //2+1] * w, axis=(0, 1))
            elif direction=='up':
                num = np.sum(self.img[x - self.k//2-1:x+self.k//2+1-1, y - self.k_w//2:y+self.k_w //2+1] * w, axis=(0, 1))
            elif direction=='left':
                num = np.sum(self.img[x - self.k//2:x+self.k//2+1, y - self.k_w//2-1:y+self.k_w //2+1-1] * w, axis=(0, 1))
            elif direction=='right':
                num = np.sum(self.img[x - self.k // 2:x + self.k // 2 + 1, y - self.k_w // 2 +1:y + self.k_w // 2 + 1 +1] * w,
                    axis=(0, 1))

            self.img[x][y] = num
        return self.img

img = np.array(Image.open('./1.jpg'))
# G = Gaosi(2.5,3)
# points = get_points()
# org_img = np.array(Image.open('./ww_org_img.png'))
# mask = G.g_mask(img,(54,1541),50,300,2)
# mask = np.expand_dims(mask,-1)
# mask = np.tile(mask,(1,1,3))
# Image.fromarray((mask*255).astype('uint8')).save('./mask.png')
# res =org_img*(mask) + (1-mask)*img
# Image.fromarray(res.astype('uint8')).save('./res.png')

# points=[]
# for i in range(42):
#     for j in range(400):
#         points.append([34+i,1500+j])
# for i in range(100,41,-1):
#     for j in range(400):
#         points.append([34+i,1500+j])

def main(mask_satrt,mask_h,mask_w):
    J = Junzhi(5,img,mask_satrt,mask_h,mask_w)
    img_save = J.k_mean()
    Image.fromarray(img_save).save('./junzhi_1.png')

mask_satrt, mask_h, mask_w = (35,1500),100,350
main(mask_satrt,mask_h,mask_w)
