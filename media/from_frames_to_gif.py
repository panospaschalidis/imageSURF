import imageio
import os, glob
from tqdm import tqdm
filenames = sorted(glob.glob(os.path.join('gif_rgb','*')))
with imageio.get_writer('gif_rgb.gif', mode='I') as writer:
    tbar = tqdm(range(len(filenames)))
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        tbar.update(1)

