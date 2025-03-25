import av
from dataset import UCF101
from models import VideoMAEModel
from transformers import AutoImageProcessor
from demo import ContinuousVideoClassifier, SlidingWindowVideoDataset, visualize

data_path = '/share/j_sun/jth264/UCF101_subset'
video_path = '/share/j_sun/jth264/UCF101_subset/test/Basketball/v_Basketball_g02_c04.avi' 

if __name__ == '__main__':
    print("Loading video...")
    container = av.open(video_path)
    print("Video loaded.")

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", use_fast = True)
    transform = lambda video: image_processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
    dataset = SlidingWindowVideoDataset(container, transform, clip_length=16, stride=5)

    classes = UCF101(data_path, load_data=False).classes
    print("Loading model...")
    model = VideoMAEModel.load_from_checkpoint("video_mae_model.ckpt", classes=classes)
    print("Model loaded.")
    classifier = ContinuousVideoClassifier(model, dataset)

    print("Running continuous classification...")
    indices, labels = classifier.run()
    print(indices)
    print(labels)
    visualize(container, indices, labels, classes)