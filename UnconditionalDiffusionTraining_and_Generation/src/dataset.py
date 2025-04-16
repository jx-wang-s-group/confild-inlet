from torch.utils.data import Dataset

def _norm(data, max_val, min_val):
    return -1 + (data - min_val)*2. / (max_val - min_val)

class Paper2DS_Re(Dataset):
    
    def __init__(self, data, wind_len, Re_list, max_val, min_val):
        super().__init__()
        self.data = _norm(data, max_val, min_val)
        self.wind_len = wind_len
        self.Re_list = Re_list
        self.nums_of_Re = len(Re_list)
        
    def __len__(self):
        return (self.data.shape[1] - self.wind_len + 1) * self.data.shape[0] 
        
    def __getitem__(self, index):
        Re_index = index % self.nums_of_Re
        win_index = index // self.nums_of_Re
        return self.data[Re_index, win_index : win_index + self.wind_len][None, ...], {"Re" : self.Re_list[Re_index]}