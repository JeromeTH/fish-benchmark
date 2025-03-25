import av
from models import CLIPImageClassifier
from transformers import AutoProcessor
from demo import ContinuousVideoClassifier, RegularFramesVideoDataset, visualize
from torchvision.datasets import Caltech101

data_path = '/share/j_sun/jth264/UCF101_subset'
video_path = '/share/j_sun/jth264/UCF101_subset/test/Basketball/v_Basketball_g02_c04.avi' 
model_checkpoint = 'clip_model.ckpt'

if __name__ == '__main__':
    container = av.open(video_path)
    
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast = True)
    transform = lambda frame: image_processor(frame, return_tensors="pt").pixel_values.squeeze(0)
    dataset = RegularFramesVideoDataset(container, transform, clip_length=16)
    # Get class names
    classes = Caltech101(".", download=True).classes    
    print("Loading model...")
    model = CLIPImageClassifier.load_from_checkpoint(model_checkpoint, classes=classes)
    print("Model loaded.")
    classifier = ContinuousVideoClassifier(model, dataset)

    print("Running continuous classification...")
    indices, labels = classifier.run()
    print(indices)
    print(labels)
    visualize(container, indices, labels, classes)