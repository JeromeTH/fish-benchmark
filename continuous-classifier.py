'''
This module continuously classifies each frame of a video clip to a certain label. If the model used is a
vision model, then it classifies each frame individually. If the model used is a video model, then it uses a 
sliding window with length of 16 frames as input and classifies the video clip as a whole. Both scenarios take in 
a video and visulize the classification results continuously.
'''
#import abstractmethod
from abc import ABC, abstractmethod
import av
import numpy as np
import cv2
from utils import read_video_pyav
from dataset import UCF101

data_path = '/share/j_sun/jth264/UCF101_subset'
video_path = '/share/j_sun/jth264/UCF101_subset/test/Basketball/v_Basketball_g02_c04.avi' 

class BaseContinuousVideoClassifier:
    '''
    Interface for continuous video classifier.
    '''
    def __init__(self, container):
        self.container = container

    @abstractmethod
    def classify(self, input):
        '''
        Classify the input.
        Args:
            Input (`np.ndarray`): Input to the model.
        Returns:
            The classification result.
        '''
        pass
    
    @abstractmethod
    def get_input_generator(self):
        '''
        Get the input generator for the video.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
        Returns:
            generator (`Generator[np.ndarray]`): Generator for inputs to the model and the corresponding frame indices.
        '''
        pass
    
    def run(self):
        '''
        Returns to vectors, one with the frame indices and the other with the predicted labels
        '''
        indices = []
        labels = []
        for index, input in self.get_input_generator():
            indices.append(index)
            labels.append(self.classify(input))
        return indices, labels

    def visualize(self, indices, labels, category_names):
        '''
        Play the video in self.container in real time and annotate each frame (defined by indices) with the predicted label.
        Args:
            indices (`List[int]`): List of frame indices.
            labels (`List[int]`): List of predicted labels.
            category_names (`List[str]`): List of category names.
        '''
        '''
    Play the video in self.container in real-time and annotate each frame (defined by indices) with the predicted label.
    
    Args:
        indices (`List[int]`): List of frame indices.
        labels (`List[int]`): List of predicted labels.
        category_names (`List[str]`): List of category names.
    '''
        container = self.container
        index_label_map = dict(zip(indices, labels))  # Map frame indices to labels
        
        for frame_idx, frame in enumerate(container.decode(video=0)):  # Decode video frames
            img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format
            
            if frame_idx in index_label_map:
                label_idx = index_label_map[frame_idx]
                label_text = category_names[label_idx] if label_idx < len(category_names) else str(label_idx)
                
                # Draw label text on frame
                cv2.putText(
                    img, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA
                )
            
            cv2.imshow('Video Classification', img)  # Display frame
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cv2.destroyAllWindows()

class VideoModelContinuousClassifier(BaseContinuousVideoClassifier):
    '''
    Classify video continuously using video model and a sliding window input generator
    '''
    def __init__(self, model, container):
        super().__init__(container)
        self.model = model
    
    def classify(self, input):
        logits = self.model(input)
        return logits.argmax()
    
    def get_input_generator(self):
        '''
        use sliding window to load video
        '''
        len = self.container.streams.video[0].frames
        clip_length = 16
        stride = 1
        for start in range(0, len - clip_length + 1, stride):
            middle_idx = start + clip_length // 2
            input = read_video_pyav(self.container, list(range(start, start + clip_length)))
            yield middle_idx, input


def load_video_batch(container):
    '''
    Yields the next 16 frames into an nd.nparray each time called
    '''
    container.seek(0)#what does this do? 
    #container.decode(video=0) returns a generator object
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        frames.append(frame)
        if i % 16 == 0:
            yield np.stack([x.to_ndarray(format="rgb24") for x in frames])
            frames = []

if __name__ == '__main__':
    #Load the video in batches of 16 frames
    container = av.open(video_path)
    classifier = VideoModelContinuousClassifier(container)
    dataset = UCF101(data_path, train=False)
    indices, labels = classifier.run()
    classifier.visualize(indices, labels, dataset.classes)