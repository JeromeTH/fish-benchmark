from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from fish_benchmark.utils import PriorityQueue
import av
import os
import numpy as np
from fish_benchmark.utils import PriorityQueue

class Behavior(BaseModel):
    '''
    A behavior in the BORIS annotation format.
    '''
    name: str
    category: str
    type: str
    def __hash__(self):
        return hash((self.category, self.name, self.type))

    def __eq__(self, other):
        return isinstance(other, Behavior) and (self.name, self.category, self.type) == (other.name, other.category, other.type)

    
class Event(BaseModel):
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    behavior: Behavior
    subject: str

class Metadata(BaseModel):
    '''
    Metadata for a BORIS annotated video.
    '''
    observation_id: str
    observation_date: str
    # observation_type: str
    source: str
    fps: float
    media_duration: float
    time_offset: float 

class BorisAnnotation:
    '''
    A BORIS annotated video and related conversion methods.
    '''
    def __init__(self, annotation_path):
        self.df = self.read_df(annotation_path)
        self.metadata: Metadata = self.parse_metadata()
        self.behaviors: List[Behavior] = self.parse_behaviors()
        self.events: List[Event] = self.parse_events()
        self.annotation_path = None
        self.video_path = None

    def read_df(self, annotation_path):
        self.annotation_path = annotation_path
        df = pd.read_csv(annotation_path)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        return df

    def parse_metadata(self):
        '''
        Load metadata from a BORIS file.
        '''
        df = self.df
        metadata = Metadata(
            observation_id=df['observation_id'][0],
            observation_date=df['observation_date'][0],
            # observation_type=df['observation_type'][0], observation type is sometimes missing, however, it's not used
            source=df['source'][0],
            fps=df['fps_(frame/s)'][0],
            media_duration=df['media_duration_(s)'][0],
            time_offset=df['time_offset_(s)'][0]
        )
        return metadata

    def parse_behaviors(self):
        '''
        Load behaviors from a BORIS file.
        '''
        df = self.df
        behaviors = df.groupby(['behavior', 'behavioral_category', 'behavior_type']).size().reset_index(name='count')
        behavior_list = []
        for _, row in behaviors.iterrows():
            behavior=Behavior(
                    name=row['behavior'],
                    category=row['behavioral_category'],
                    type=row['behavior_type']
                )
            behavior_list.append(behavior)
        return behavior_list

    def parse_events(self):
        '''
        Load events from a BORIS file.
        '''
        df = self.df
        events = []
        for _, row in df.iterrows():
            event = Event(
                start_time=row['start_(s)'],
                end_time=row['stop_(s)'],
                start_frame=self.to_frame(row['start_(s)']),
                end_frame=self.to_frame(row['stop_(s)']),
                behavior=Behavior(
                    name=row['behavior'],
                    category=row['behavioral_category'],
                    type=row['behavior_type']
                ),
                subject=row['subject']
            )
            events.append(event)
        return events

    def load_video(self, video_path):
        '''
        Load a video from a path.
        '''
        self.video_path = video_path
        self.container = av.open(video_path)
        self.frame_count = self.container.streams.video[0].frames
        self.check_fps_match()

    def check_fps_match(self):
        annotated_frame_count = int(self.metadata.media_duration * self.metadata.fps)
        #1 frame error margin
        if abs(annotated_frame_count - self.frame_count) > 1:
            raise ValueError(f"frames mismatch: expected {annotated_frame_count} but got {self.frame_count}")
    
    def to_frame(self, timestamp):
        '''
        Convert a timestamp to a frame number.
        '''
        return int((timestamp - self.metadata.time_offset)* self.metadata.fps)
    
    def to_timestamp(self, frame):
        '''
        Convert a frame number to a timestamp.
        '''
        return frame / self.metadata.fps + self.metadata.time_offset
    
    def stream_frame_annotations(self):
        '''
        Stream the video frames and their annotations.
        Algorithm: 
        Sort the events by start time
        maintain the set of active events with a priority queue, priority determined by the end time of each event
        For each frame in the video:
            Check if the next closest event should start.
            If so, add it to the active events
            Annotate the frame with the active events
            check if the top most event should end
            If so, remove it from the active events
        
        Yields a tuple a training example compatible with the webdataset format. 
        e.g.
            <video_name>_<frame_id>.json #the annotation and metadata
            <video_name>_<frame_id>.png #the video frame
        '''
        #put events in a priority queue
        event_queue = PriorityQueue()
        for id, event in enumerate(self.events):
            event_queue.push((event.start_frame,id, event))
        # Initialize the priority queue
        active_events = PriorityQueue()
        for frame_id, frame in enumerate(self.container.decode(video=0)):
            # Check if any events should start
            while not event_queue.is_empty() and event_queue.peek()[0] <= frame_id:
                _, id, event = event_queue.pop()
                active_events.push((event.end_frame, id, event))
            # Annotate the frame with the active events
            annotations = {'events': [], 
                           'metadata': self.metadata.model_dump(),
                           'frame_id': frame_id, 
                           'video_path': self.video_path}
            # annotations should be a list of dictionaries which would later be converted to json
            # in the webdataset format
            for _, _, event in active_events.to_list():
                annotations['events'].append(event.model_dump())

            yield frame.to_image(), annotations
            # Check if any events should end
            while not active_events.is_empty() and active_events.peek()[0] <= frame_id:
                active_events.pop()
