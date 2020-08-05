import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PionsDataset(Dataset):
    
    def __init__(self, path_list):
        self.path_list = path_list
        part = np.array([])
        index = np.array([])
        for i,path in enumerate(path_list):
            file = h5py.File(path, 'r')['hcal_only/energy']
            part = np.append(part, np.ones(len(file))*i)
            index = np.append(index, np.arange(len(file)))
            
                
#         self.keys = pd.DataFrame({'part' : part.astype(int),
#                                   'index' : index.astype(int)
#                                  })
        
        self.keys = np.vstack([part.astype(int), index.astype(int)]).T

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
#         part = self.keys['part'][idx]
#         index = self.keys['index'][idx]
        part = self.keys[idx][0]
        index = self.keys[idx][1]
        path = self.path_list[part]
        
        file = h5py.File(path, 'r')['hcal_only']
        energy = file['energy'][index]
        shower = file['layers'][index]
        
        energy = energy.reshape(1,1,1,1)
        shower = np.expand_dims(shower, 0)

        return {'energy' : energy,
                'shower' : shower}