from image import ImageReader
from typing import List
import torch
import time
from extension import get_image_to_polygon_module

class Tracer:
    def __init__(self, 
                 name : str,
                 device_id : int = None,
                 ) -> None:
        self.name = name
        self.device_id = device_id
        if self.device_id is not None:
            assert(torch.cuda.is_available())
        self.GPU = torch.device(f"cuda:{self.device_id}" if self.device_id is not None else "cpu")
    
    def getImage(self) -> torch.Tensor:
        reader = ImageReader(self.name, self.GPU)
        return reader.getImage()
    
    def trace(self,
              image : torch.Tensor,
              GPU=True) -> List[torch.Tensor]:
        if GPU:
            image_to_polygon = get_image_to_polygon_module()
            start_time = time.time()
            ptr, polygon =  image_to_polygon.image_to_polygon_cuda(image)
            end_time = time.time()
            return [ptr.cpu(), polygon.cpu(), end_time - start_time]
            
        else:
            image_to_polygon = get_image_to_polygon_module()
            start_time = time.time()
            image_cpu = image.cpu()
            ptr, polygon = image_to_polygon.image_to_polygon_cpu(image_cpu)
            end_time = time.time()
            return [ptr, polygon, end_time - start_time]
        
    
    
        
