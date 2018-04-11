import network
import random,os,logging,time
import model_detect_data
import tensorflow as tf
import numpy as np
import model_comm_meta as meta
import tensorflow.contrib.slim as slim

from scipy import misc

# 总迭代次数
TRAINING_STEPS = 1000000
# 初始学习率
LEARNING_RATE_BASE = 0.00001
# 衰减比例
DECAY_RATE = 0.9
DECAY_STAIRCASE = True
# 衰减步数
DECAY_STEPS = 2000
MOMENTUM = 0.9
# 候选区域高度
anchor_heights = [12, 24, 36, 48, 64, 80, 96]
# 模型地址
CKPT_PATH = 'models/resnet_v2_50/resnet_v2_50.ckpt'
# 显存保留
GPU_MEMORY_KEEP = 0.8
# 不训练的scope
not_train_vars_scope = ['resnet_v2_50/conv1','resnet_v2_50/block1','resnet_v2_50/block2','resnet_v2_50/block3']



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Model():

    def __init__(self):

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_KEEP



    def define_graph(self, graph):
        with graph.as_default():

            x = tf.placeholder(tf.float32, [1,None,None,3], name='x-input')
            sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')

            conv_feat = network.resnet_layers(input=x,output_stride=8)

            rnn_cls, rnn_ver, rnn_hor = network.rnn_detect_layers(conv_feat,
                                                                  sequence_length,
                                                                  len(anchor_heights))

            t_cls = tf.placeholder(tf.float32, (None, None, None), name = 'c-input')
            t_ver = tf.placeholder(tf.float32, (None, None, None), name = 'v-input')
            t_hor = tf.placeholder(tf.float32, (None, None, None), name = 'h-input')

            loss = network.detect_loss(rnn_cls, rnn_ver, rnn_hor, t_cls, t_ver, t_hor)

            vars_to_train = network.get_variables_to_train(exclude=not_train_vars_scope)

            global_step = tf.train.get_or_create_global_step()

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(extra_update_ops):

                learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           tf.train.get_global_step(),
                                                           DECAY_STEPS,
                                                           DECAY_RATE,
                                                           staircase = DECAY_STAIRCASE,
                                                           name = 'learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                   beta1 = MOMENTUM,name='train_op').minimize(loss,global_step=global_step,var_list=vars_to_train)

    def save(self,step,sess, path='models/ctpn/'):
        saver = tf.train.Saver(tf.trainable_variables())
        now = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        path = path + now + '-' + str(step) + '/' + 'model.ckpt'
        os.makedirs(path)
        saver.save(sess, path)
        logging.info("保存成功！")


    def restore(self, sess, ckpt_path):

        restorer = tf.train.Saver(self.graph.get_collection('model_variables'))
        restorer.restore(sess, ckpt_path)


    def get_node(self):

        self.x = self.graph.get_tensor_by_name('x-input:0')
        self.t_cls = self.graph.get_tensor_by_name('c-input:0')
        self.t_ver = self.graph.get_tensor_by_name('v-input:0')
        self.t_hor = self.graph.get_tensor_by_name('h-input:0')
        self.rnn_cls = self.graph.get_tensor_by_name('RNN_module/rnn_cls:0')
        self.rnn_ver = self.graph.get_tensor_by_name('RNN_module/rnn_ver:0')
        self.rnn_hor = self.graph.get_tensor_by_name('RNN_module/rnn_hor:0')
        self.loss = self.graph.get_tensor_by_name('loss:0')
        self.train_op = self.graph.get_operation_by_name('train_op')
        self.global_step = self.graph.get_tensor_by_name('global_step:0')
        self.learning_rate = self.graph.get_tensor_by_name('learning_rate:0')
        self.conv_feat = self.graph.get_tensor_by_name('conv_feat:0')
        self.sequence_length = self.graph.get_tensor_by_name('sequence_length:0')




    def train(self, data_path='validation/',):

        list_path = model_detect_data.getFilesInDirect(data_path)

        self.graph = tf.Graph()
        self.define_graph(self.graph)


        with self.graph.as_default():
            with tf.Session(config=self.config) as sess:

                self.get_node()
                self.restore(sess,ckpt_path=CKPT_PATH)
                tf.global_variables_initializer().run()



                total_loss = 0
                for i in range(1, TRAINING_STEPS):

                    img_file = random.choice(list_path)
                    img_data, rate = model_detect_data.transform_image(img_file)

                    feat = sess.run(self.conv_feat,feed_dict={self.x: [img_data],})
                    feat_size = (feat.shape[0], feat.shape[1])

                    img_data, target_cls, target_ver, target_hor = \
                    model_detect_data.getImageAndTargets(img_file, anchor_heights, feat_size)

                    feed_dict = {self.x: img_data,
                                 self.t_cls: target_cls,
                                 self.t_ver: target_ver,
                                 self.t_hor: target_hor,
                                 self.sequence_length:np.ones([feat_size[0]])*64}



                    _, loss_value, step, lr = sess.run(
                        [   self.train_op,
                            self.loss,
                            self.global_step,
                            self.learning_rate,
                        ],  feed_dict)

                    total_loss += loss_value
                    display = 500
                    if  i % display == 0:
                        logging.info('iter: {:}, loss: {:.4f}, learning_rate: {:g}'.format(step,total_loss/display,lr))
                        # print('step: %d, loss: %g, lr: %g, ' %
                        #       (step, total_loss/50, lr))
                        total_loss = 0

                    if i in [50000, 60000, 70000, 80000, 90000]:
                        self.save(sess=sess,step=i)

                    if  i % 5000 == 0:
                        path = 'result/' + str(i) + '/'
                        if not os.path.exists(path): os.mkdir(path)
                        show_list = model_detect_data.getFilesInDirect('result/')
                        count = 0
                        for show_path in show_list:
                            img_data, rate = model_detect_data.transform_image(show_path)
                            feat = sess.run(self.conv_feat, feed_dict={
                                self.x: [img_data],
                            })
                            feat_size = (feat.shape[0], feat.shape[1])

                            img_data, target_cls, target_ver, target_hor = model_detect_data.getImageAndTargets(show_path, meta.anchor_heights, feat_size)


                            feed_dict = {self.x: img_data,
                                         self.t_cls: target_cls,
                                         self.t_ver: target_ver,
                                         self.t_hor: target_hor,
                                         self.sequence_length: np.ones([feat_size[0]]) * 64}

                            r_cls, r_ver, r_hor = sess.run([self.rnn_cls, self.rnn_ver, self.rnn_hor], feed_dict)

                            file_target = path + str(count) + '.png'
                            count += 1
                            img, rate = model_detect_data.transform_image(show_path)
                            misc.imsave(file_target, img)
                            # trans
                            text_bbox = model_detect_data.transResults(r_cls, r_ver, r_hor, anchor_heights, meta.threshold)
                            #
                            model_detect_data.drawTextBox(file_target, text_bbox)




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
    model = Model()
    model.train()




