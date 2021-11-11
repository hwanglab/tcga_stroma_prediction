###############################################################################
#  Copyright Sanghoon Lee leesan@marshall.edu
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

"""
Network class

Initialize U-net network model.
Perform train model.
Return predicted probabilities/Predicted labels.
"""

import numpy as np
import tensorflow as tf

class Network():

    def __init__(self):

        # set variables
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 128
        self.IMG_CHANNELS = 3

        self.input_units = 64
        self.hidden_units = 32
        self.output_units = 1
        self.epochs = 100
        self.dropout = 0.3
        self.activation = 'relu'
        self.kernel = 'he_normal'
        self.padding = 'same'
        self.activation_last = 'sigmoid'
        self.optimizer = 'Adam'
        self.learning_rate = 0.001
        self.loss = 'binary_crossentropy'
        self.noise_shape = None
        self.seed = 145
        self.batch_size = 32
        self.metrics = 'accuracy'
        self.model = None
        self.classifier = None

    def init_model(self):

        # build u-net model
        inputs = tf.keras.layers.Input((self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS))
        s = tf.keras.layers.Lambda(lambda x: x / 255) (inputs)

        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (s)
        c1 = tf.keras.layers.Dropout(0.1) (c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2)) (c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (p1)
        c2 = tf.keras.layers.Dropout(0.1) (c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (p2)
        c3 = tf.keras.layers.Dropout(0.2) (c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (p3)
        c4 = tf.keras.layers.Dropout(0.2) (c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (p4)
        c5 = tf.keras.layers.Dropout(0.3) (c5)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c5)

        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=self.padding) (c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (u6)
        c6 = tf.keras.layers.Dropout(0.2) (c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=self.padding) (c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (u7)
        c7 = tf.keras.layers.Dropout(0.2) (c7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=self.padding) (c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (u8)
        c8 = tf.keras.layers.Dropout(0.1) (c8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding=self.padding) (c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (u9)
        c9 = tf.keras.layers.Dropout(0.1) (c9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=self.activation, kernel_initializer=self.kernel, padding=self.padding) (c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=self.activation_last) (c9)

        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])

    def loading_model(self, path):

        self.model = tf.keras.models.load_model(path)

    def saving_model(self, path):

        self.model.save(path)

    def train_model(self, features, labels):

        self.model.fit(features, labels, batch_size=self.batch_size, epochs=self.epochs)

    def predict_prob(self, features):

        predicts_prob = self.model.predict(features)
        return predicts_prob

    def predict(self, features):

        predicts_prob = self.model.predict(features)
        predicts = (predicts_prob > 0.5).astype(np.bool)
        return predicts
