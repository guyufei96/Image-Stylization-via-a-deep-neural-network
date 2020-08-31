import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os


# input and output size of images
PIC_ROW = 800
PIC_COL = 600
Content_Dir =  './Images/Content/Taipei101.jpg'	#input content image direction
Style_Dir = './Images/Style/StarryNight.jpg'	#input style image direction
Output_Dir = './results'	                #output image direction
Output_Img = 'results.jpg'	                #output image name
VGG_Path = 'imagenet-vgg-verydeep-19.mat'	#input path of VGG19
Noise = 0.7		#noise ratio
ALPHA = 1		#content strength
BETA = 500		#noise strength
Iteration = 5000	# iteration times

def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                  strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i,):
  weights = vgg_layers[i][0][0][0][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][0][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias

def build_vgg19(path):
  net = {}
  vgg_rawnet = scipy.io.loadmat(path)
  vgg_layers = vgg_rawnet['layers'][0]
  net['input'] = tf.Variable(np.zeros((1, PIC_COL, PIC_ROW, 3)).astype('float32'))
  net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
  net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
  net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
  net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
  net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
  net['pool3']   = build_net('pool',net['conv3_4'])
  net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19))
  net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
  net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
  net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
  net['pool4']   = build_net('pool',net['conv4_4'])
  net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28))
  net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
  net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
  net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
  net['pool5']   = build_net('pool',net['conv5_4'])
  return net

def Content_loss(p, x):
  M = p.shape[1]*p.shape[2]
  N = p.shape[3]
  loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))  
  return loss


def Style_loss(a, x):
  M = a.shape[1]*a.shape[2]
  N = a.shape[3]
  x1 = a.reshape(M,N)
  A = np.dot(x1.T, x1)
  x2 = tf.reshape(x,(M,N))
  G = tf.matmul(tf.transpose(x2), x2)
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
  return loss


def main():
  net = build_vgg19(VGG_Path)
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  noise_img = np.random.uniform(-20, 20, (1, PIC_COL, PIC_ROW, 3)).astype('float32')
  
  image = scipy.misc.imread(Content_Dir)
  image = scipy.misc.imresize(image,(PIC_COL,PIC_ROW))
  image = image[np.newaxis,:,:,:] 
  content_img = image - (np.array([123, 117, 104]).reshape((1,1,1,3))) 	# input content image

  image = scipy.misc.imread(Style_Dir)
  image = scipy.misc.imresize(image,(PIC_COL,PIC_ROW))
  image = image[np.newaxis,:,:,:] 
  style_img = image - (np.array([123, 117, 104]).reshape((1,1,1,3)))   # input style image
  
  #build content loss from content layer 4-2 
  content_loss = lambda layer : layer[1]*Content_loss(sess.run(net[layer[0]]) ,  net[layer[0]])
  sess.run([net['input'].assign(content_img)])
  cost_content = sum(map(content_loss , [('conv4_2',1.)]))	
  #build style loss from style layers[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)])
  style_loss = lambda l: l[1]*Style_loss(sess.run(net[l[0]]) ,  net[l[0]])
  sess.run([net['input'].assign(style_img)])
  cost_style = sum(map(style_loss , [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]))	

  cost_total = ALPHA * cost_content + BETA * cost_style
  learning = tf.train.AdamOptimizer(2.0).minimize(cost_total)	#Adam optimization algorithm: the quadratic gradient correction is introduced in the global optimal optimization algorithm

  sess.run(tf.initialize_all_variables())
  sess.run(net['input'].assign( Noise* noise_img + (1.-Noise) * content_img))

  if not os.path.exists(Output_Dir):
      os.mkdir(Output_Dir)

  for i in range(Iteration):
    sess.run(learning)
    print(i)
    if i%100 ==0:
      result_img = sess.run(net['input'])  
      result_img = result_img + (np.array([123, 117, 104]).reshape((1,1,1,3)))
      result_img = result_img[0]
      result_img = np.clip(result_img, 0, 255).astype('uint8')
      scipy.misc.imsave(os.path.join(Output_Dir,'Iteration_%s.jpg'%(str(i).zfill(4))), result_img)  #each 100 iteration output learning image
      
  scipy.misc.imsave(os.path.join(Output_Dir,Output_Img),result_img)	#after 5000 iteration training, generate the output image

if __name__ == '__main__':
  main()
