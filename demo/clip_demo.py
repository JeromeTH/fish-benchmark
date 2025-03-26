import av
from fish_benchmark.models import CLIPImageClassifier
from transformers import AutoProcessor
from fish_benchmark.demo import ContinuousVideoClassifier, RegularFramesVideoDataset, visualize
from torchvision.datasets import Caltech101
from torch.utils.data import DataLoader

video_path = '/share/j_sun/jth264/UCF101_subset/test/Basketball/v_Basketball_g02_c04.avi' 
model_checkpoint = 'clip_model.ckpt'
output_path = 'clip_demo.mp4'

if __name__ == '__main__':
    container = av.open(video_path)
    
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    transform = lambda frame: image_processor(images = frame, return_tensors="pt").pixel_values.squeeze(0)
    dataset = RegularFramesVideoDataset(container, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # Get class name
    print("Loading model...")
    classes = Caltech101(".", download=True).categories    
    model = CLIPImageClassifier.load_from_checkpoint(model_checkpoint, num_classes=len(classes))
    print("Model loaded.")
    classifier = ContinuousVideoClassifier(model, dataloader)

    print("Running continuous classification...")
    indices, labels = classifier.run()
    print(indices)
    print(labels)
    visualize(container, indices, labels, classes, output_path=output_path)