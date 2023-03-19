# sd-video

Text to Video


Example:
```py
from sd_video import SDVideo, save_vid
model = SDVideo('/path/to/model_and_config', 'cuda')
x = model('arnold schwarzenegger eating a giant cheeseburger')
save_vid(x, 'output') # 0001.png ... 00NN.png
```

![](examples/arnold_burger.gif)
