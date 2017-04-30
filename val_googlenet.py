from PIL import Image
import numpy as np
import chainer
import chainer.functions as F
from chainer.functions import caffe

class NNmodel(object):
    def __init__(self):
        self.image_shape = self._image_shape()
        self.mean_image = self._mean_image()
        self.func = None
        self.categories = None

    def _image_shape(self):
        raise NotImplementedError

    def _mean_image(self):
        raise NotImplementedError

    def _predict_class(self):
        raise NotImplementedError

    def load(self,path):
        self.func = caffe.CaffeFunction(path)

    def load_label(self, path):
        self.categories = np.loadtxt(path, str, delimiter="\n")

    def print_prediction(self, image_path, rank=3):
        prediction = self.predict(image_path, rank)
        for i, (score, label)in enumerate(prediction[:rank]):
            print '{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label)

    def predict(self,image_path, rank=3):
        x = chainer.Variable(self.load_image(image_path), volatile=True)
        y = self._predict_class(x)
        result = zip(y.data.reshape((y.data.size)), self.categories)
        return sorted(result, reverse=True)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        w_in, h_in = self.image_shape
        w, h = image.size

        if w > h:
            shape = (w_in * w / h, h_in)
        else:
            shape = (w_in, h_in * h / w)

        x = (shape[0] - w_in) / 2
        y = (shape[1] - h_in) / 2
        pixels = np.asarray(image.resize(shape).crop((x, y, x + w_in, y + h_in ))).astype(np.float32)
        pixels = pixels[:,:,::-1].transpose(2,0,1)
        pixels -= self.mean_image
        return pixels.reshape((1,) + pixels.shape)


class GoogleNet(NNmodel):
    def __init__(self):
        NNmodel.__init__(self)

    def _image_shape(self):
        return (224, 224)

    def _mean_image(self):
        mean_image = np.ndarray((3, 224, 224), dtype=np.float32)
        mean_image[0] = 103.939
        mean_image[1] = 116.779
        mean_image[2] = 123.68
        return mean_image

    def _predict_class(self, x):
        y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'], disable=['loss1/ave_pool', 'loss2/ave_pool'], train=False)
        return F.softmax(y)