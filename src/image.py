import PIL
from PIL import Image
import torch
from utils import getBenchmarkPath

class ImageReader:
    def __init__(self, 
                 name : str, 
                 device : torch.device) -> None:
        self.name = name
        self.benchmark_path = getBenchmarkPath()
        self.image_path = self.benchmark_path + f"/{self.name}.png"
        
        self.device : torch.device = device
        
    def getImage(self) -> torch.Tensor:
        try:
            image = Image.open(self.image_path).convert('L')
        except Exception as e:
            PIL.Image.MAX_IMAGE_PIXELS = 933120000
            image = Image.open(self.image_path).convert('L')
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = data.reshape(image.height, image.width).float().div(255.0)
        tensor = (tensor > 0.5).float().squeeze()
        if self.device.type == "cpu":
            return tensor
        else:
            return tensor.to(self.device)
        
