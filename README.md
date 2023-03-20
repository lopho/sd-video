# sd-video

Text to Video


## Example
```py
from sd_video import SDVideo, save_vid
model = SDVideo('/path/to/model_and_config', 'cuda')
x = model('arnold schwarzenegger eating a giant cheeseburger')
save_vid(x, 'output') # 0001.png ... 00NN.png
```

![](examples/arnold_burger.gif)


## Sampling options
```py
model(
  text = 'some text', # text conditioning
  text_neg = 'other text' # negative text conditioning
  guidance_scale = 9.0, # positive / negative conditioning ratio (cfg)
  timesteps = 50, # sampling steps
  image_size = (256, 256), # output image resolution (w,h)
  num_frames = 16, # number of video frames to generate
  eta = 0.0, # DDIM randomness
  bar = False, # display TQDM progress bar for sampling process
)
```
  
  
