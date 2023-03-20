# sd-video

Text to Video


Example:
```py
from sd_video import SDVideo, save_gif
model = SDVideo('/path/to/model_and_config', 'cuda')
x = model('arnold schwarzenegger eating a giant cheeseburger')
save_gif(x, 'output.gif')
```

![](examples/arnold_burger.gif)
