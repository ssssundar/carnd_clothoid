import numpy as np
from styx_msgs.msg import TrafficLight
import tensorflow as tf

class TLClassifier(object):
    def __init__(self, is_site):
        classifier_model = 'light_classification/classifiers/sim_model_classifier/frozen_inference_graph.pb'
        if is_site:
            classifier_model = 'light_classification/classifiers/real_model_classifier/frozen_inference_graph.pb'
        # TODO: experiment with different thresholds to pick the best
        self.threshold = .5

        self.graph = self._load_graph(classifier_model)
        print('graph loaded')
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.graph)

    def _load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
 
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        return graph
 

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (scores, classes) = self.sess.run(
                [self.scores, self.classes],
                feed_dict={self.image_tensor: image_expanded})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        print('scores: ', scores[0])
        print('classes: ', classes[0])

        if (len(scores) > 0) and (scores[0] > self.threshold):
            if classes[0] == 1:
                return TrafficLight.GREEN
            if classes[0] == 2:
                return TrafficLight.RED
            if classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
