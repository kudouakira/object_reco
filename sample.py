from val_googlenet import GoogleNet

m = GoogleNet()
m.load("bvlc_googlenet.caffemodel")
m.load_label("labels.txt")
m.print_prediction("kanekin.jpg")