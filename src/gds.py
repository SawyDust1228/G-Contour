from gdspy import GdsLibrary
from gdspy import PolygonSet

class GDSWriter:
    def __init__(self,
                 name : str,
                 folder_path : str,
                 GPU=True) -> None:
        self.name = name
        self.folder_path = folder_path
        self.GPU = GPU
        
    def _getPolygons(self, ptr, polygons):
        return [polygons[ptr[i]:ptr[i + 1]] for i in range(ptr.shape[0] - 1)]
    
    def write(self, ptr, polygons):
        library = GdsLibrary(name=self.name, unit=1e-6, precision=1e-9)
        top = library.new_cell(name=self.name)
        polygons_np = list(map(lambda x : x.numpy() / 1e3, self._getPolygons(ptr, polygons)))
        top.add(PolygonSet(polygons_np))        
        file_name = self.name + "_gpu" if self.GPU else self.name
        library.write_gds(f"{self.folder_path}/{file_name}.gds")
    
        
        
