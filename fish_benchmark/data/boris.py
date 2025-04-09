from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from fish_benchmark.utils import PriorityQueue
import av
import os
import numpy as np
from fish_benchmark.utils import PriorityQueue
from fish_benchmark.data.schemas import Behavior, Event, Metadata

BORIS_NAME_TO_METADATA = {
    'observation_id': 'observation_id',
    'observation_date': 'observation_date',
    # 'observation_type': 'observation_type', observation type is sometimes missing, however, it's not used
    'source': 'source',
    'fps_(frame/s)': 'fps',
    'media_duration_(s)': 'media_duration',
    'time_offset_(s)': 'time_offset'
}

BORIS_NAME_TO_EVENT = {
    'start_(s)': 'start_time',
    'stop_(s)': 'end_time',
    'subject': 'subject',
}

BORIS_NAME_TO_BEHAVIOR = {
    'behavior': 'name',
    'behavioral_category': 'category',
    'behavior_type': 'type'
}

def read_df(annotation_path):
    df = pd.read_csv(annotation_path)
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

def parse_behaviors(df):
    '''
    Load behaviors from a BORIS file.
    '''
    behaviors = df.groupby([boris_col_name for boris_col_name in BORIS_NAME_TO_BEHAVIOR.keys()]).size().reset_index(name='count')
    behavior_list = []
    for _, row in behaviors.iterrows():
        behavior_list.append(
            Behavior(**{
                    metadata_field: row[boris_col]
                    for boris_col, metadata_field in BORIS_NAME_TO_BEHAVIOR.items()
                })
            )
    return behavior_list

class BorisAnnotation:
    '''
    A BORIS annotated video and related conversion methods.
    '''
    def __init__(self, annotation_path):
        self.annotation_path = annotation_path
        self.df = read_df(self.annotation_path)
        self.metadata: Metadata = Metadata(**{
            metadata_field: self.df[boris_col][0]
            for boris_col, metadata_field in BORIS_NAME_TO_METADATA.items()
        })
        self.behaviors: List[Behavior] = parse_behaviors(self.df)
        self.events: List[Event] = self.parse_events(self.df)
        self.annotation_path = None
        self.video_path = None

    def parse_events(self, df):
        '''
        Load events from a BORIS file.
        '''
        events = []
        for _, row in df.iterrows():
            event_kwargs = {
                event_field: row[boris_col]
                for boris_col, event_field in BORIS_NAME_TO_EVENT.items()
            }

            event_kwargs.update({
                'behavior': Behavior(**{
                    metadata_field: row[boris_col]
                    for boris_col, metadata_field in BORIS_NAME_TO_BEHAVIOR.items()
                }),
                'start_frame': self.to_frame(row['start_(s)']),
                'end_frame': self.to_frame(row['stop_(s)'])
            })

            event = Event(**event_kwargs)
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
