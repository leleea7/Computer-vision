import os
import data_preprocessing as dp
import tensorflow as tf
import evaluation as ev
import plot_generator as pg
import pickle
import tqdm
from Xception import Xception
from sklearn.model_selection import train_test_split


class EmotionDetector:

    def __init__(self):
        self.__BATCH_SIZE = 32
        self.__IMG_SIZE = (48, 48, 1)
        self.__DATA_DIR = 'challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/'
        self.__TMP_DIR = 'tmp/'
        self.__SENTIMENTS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.__xtrain = []
        self.__ytrain = []
        self.__xtest = []
        self.__ytest = []

        if not os.path.exists(self.__TMP_DIR):
            os.makedirs(self.__TMP_DIR)

        self.__GLOBAL_EPOCH = dp.global_epoch(self.__TMP_DIR + 'epoch.txt')
        print('Global epoch:', self.__GLOBAL_EPOCH)

        try:
            f = open(self.__TMP_DIR + 'train_len.txt', 'r', encoding='utf8')
            self.__xtrain_len = int(f.read())
            f.close()
        except:
            self.__load_dataset(notify=True)
            f = open(self.__TMP_DIR + 'train_len.txt', 'w', encoding='utf8')
            f.write(str(len(self.__flow) * self.__BATCH_SIZE))
            f.close()
            self.__xtrain_len = len(self.__flow) * self.__BATCH_SIZE

            self.__xtrain = None
            self.__ytrain = None
            self.__xtest = None
            self.__ytest = None
            self.__faces = None
            self.__emotions = None
            self.__data_generator = None
            self.__flow = None


        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__images = tf.placeholder(tf.float32, shape=[None, self.__IMG_SIZE[0], self.__IMG_SIZE[1], self.__IMG_SIZE[2]])
            self.__labels = tf.placeholder(tf.int32, shape=[None])

            output = Xception(self.__images, len(self.__SENTIMENTS))

            with tf.name_scope('loss'):
                self.__losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.__labels)
                self.__loss = tf.reduce_mean(self.__losses)

            with tf.name_scope('optimizer'):
                self.__global_step = tf.Variable(0, trainable=False)
                self.__lr = tf.train.exponential_decay(learning_rate=0.001, global_step=self.__global_step * self.__BATCH_SIZE,
                                                       decay_steps=self.__xtrain_len, decay_rate=0.95, staircase=True)
                self.__optimizer = tf.train.AdamOptimizer(learning_rate=self.__lr)
                self.__train_op = self.__optimizer.minimize(self.__loss, global_step=self.__global_step)

            with tf.name_scope('prediction'):
                self.__labels_predicted = tf.argmax(output, axis=1)

            self.__saver = tf.train.Saver()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            ### SESSION ###
            self.__session = tf.Session(graph=self.__graph)


            # We must initialize all variables before we use them.
            init.run(session=self.__session)

            # reload the model if it exists and continue to train
            try:
                self.__saver.restore(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
                print('Model restored')
            except:
                print('Model initialized')

    def __load_dataset(self, notify=False):
        if os.path.exists(self.__TMP_DIR + 'xtrain.pkl') and os.path.exists(self.__TMP_DIR + 'ytrain.pkl') and os.path.exists(
                self.__TMP_DIR + 'xtest.pkl') and os.path.exists(self.__TMP_DIR + 'ytest.pkl'):
            if notify:
                print('Reloading train set and test set...')
            self.__xtrain = pickle.load(open(self.__TMP_DIR + 'xtrain.pkl', 'rb'))
            self.__ytrain = pickle.load(open(self.__TMP_DIR + 'ytrain.pkl', 'rb'))
            self.__xtest = pickle.load(open(self.__TMP_DIR + 'xtest.pkl', 'rb'))
            self.__ytest = pickle.load(open(self.__TMP_DIR + 'ytest.pkl', 'rb'))
            if notify:
                print('Train set and test set reloaded')
        else:
            if notify:
                print('Generating train set and test set...')
            self.__faces, self.__emotions = dp.load_fer2013(self.__DATA_DIR)
            self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(self.__faces, self.__emotions,
                                                                                        test_size=0.2,
                                                                                        shuffle=True)
            pickle.dump(self.__xtrain, open(self.__TMP_DIR + 'xtrain.pkl', 'wb'))
            pickle.dump(self.__ytrain, open(self.__TMP_DIR + 'ytrain.pkl', 'wb'))
            pickle.dump(self.__xtest, open(self.__TMP_DIR + 'xtest.pkl', 'wb'))
            pickle.dump(self.__ytest, open(self.__TMP_DIR + 'ytest.pkl', 'wb'))
            if notify:
                print('Train set and test set generated and stored in tmp folder')

        self.__data_generator = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                                                featurewise_std_normalization=False,
                                                                                rotation_range=10,
                                                                                width_shift_range=0.1,
                                                                                height_shift_range=0.1,
                                                                                zoom_range=0.1,
                                                                                horizontal_flip=True)

        self.__flow = self.__data_generator.flow(self.__xtrain, self.__ytrain, self.__BATCH_SIZE)

    def train(self, epochs=10):
        if self.__xtrain and self.__ytrain and self.__xtest and self.__ytest:
            pass
        else:
            self.__load_dataset()

        # Open a writer to write summaries.
        self.__writer = tf.summary.FileWriter(self.__TMP_DIR, self.__session.graph)

        for epoch in range(epochs):
            #learning_rate = self.__session.run(self.__lr)
            #print('Learning rate', learning_rate)

            average_loss = 0
            num_steps = len(self.__flow)

            for step in tqdm.tqdm(range(num_steps), desc='Epoch ' + str(epoch + 1 + self.__GLOBAL_EPOCH) + '/' + str(epochs + self.__GLOBAL_EPOCH)):

                batch, label = self.__flow.next()

                run_metadata = tf.RunMetadata()
                _, l = self.__session.run([self.__train_op, self.__loss], feed_dict={self.__images: batch, self.__labels: label},
                                          run_metadata=run_metadata)

                average_loss += l

                # print loss and accuracy on test set at the and of each epoch
                if step == num_steps - 1:

                    y_true = []
                    y_pred = []

                    for i in range(len(self.__xtest)):
                        prediction = self.__session.run(self.__labels_predicted, feed_dict={self.__images: [self.__xtest[i]]},
                                                        run_metadata=run_metadata)

                        y_true.append(self.__ytest[i])
                        y_pred.append(prediction[0])

                    accuracy = ev.accuracy(y_true, y_pred)

                    print('Loss:', str(average_loss / step), '\tAccuracy:', accuracy)

                    with open(self.__TMP_DIR + '/log.txt', 'a', encoding='utf8') as f:
                        f.write(str(accuracy) + ' ' + str(average_loss / step) + '\n')

                if step == (num_steps - 1) and epoch + 1 == epochs:
                    s = self.__session.run(self.__global_step)
                    self.__writer.add_run_metadata(run_metadata, 'step%d' % step, global_step=s)

        self.__saver.save(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
        dp.global_epoch(self.__TMP_DIR + 'epoch.txt', update=self.__GLOBAL_EPOCH + epochs)

        self.__writer.close()

        pg.generate_accuracy_plot(data_dir=self.__TMP_DIR)
        pg.generate_loss_plot(data_dir=self.__TMP_DIR)

        conf_mat = ev.confusion_matrix(y_true, y_pred, len(self.__SENTIMENTS))
        pg.generate_confusion_matrix_plot(conf_mat, self.__SENTIMENTS, data_dir=self.__TMP_DIR)
        pg.generate_confusion_matrix_plot(conf_mat, self.__SENTIMENTS, normalize=True, data_dir=self.__TMP_DIR)

    def predict(self, images):
        run_metadata = tf.RunMetadata()
        prediction = self.__session.run(self.__labels_predicted, feed_dict={self.__images: images},
                                        run_metadata=run_metadata)

        return prediction
