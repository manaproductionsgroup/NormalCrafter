## ___***NormalCrafter: Learning Temporally Consistent Video Normal from Video Diffusion Priors***___

## this is a slightly modified version to fit with Mana Productions needs (the changes are extracting the unique frames from vidoe output with the frame number as filename)

<div align="center">
<br>

_**[Yanrui Bin<sup>1</sup>](https://binyr.github.io/),[Wenbo Hu<sup>2*](https://wbhu.github.io), 
[Haoyuan Wang<sup>3](https://www.whyy.site/), 
[Xinya Chen<sup>4](https://xinyachen21.github.io/), 
[Bing Wang<sup>2 &dagger;</sup>](https://bingcs.github.io/)**_
<br>
<sup>1</sup>Spatial Intelligence Group, The Hong Kong Polytechnic University
<sup>2</sup>ARC Lab, Tencent PCG
<sup>3</sup>City University of Hong Kong
<sup>4</sup>Huazhong University of Science and Technology
<!-- </div> -->

ICCV 2025

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2504.11427'><img src='https://img.shields.io/badge/arXiv-2504.01016-b31b1b.svg'></a> &nbsp;
 <a href='https://normalcrafter.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/spaces/Yanrui95/NormalCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;

</div>

## 🔆 Notice
We recommend that everyone use English to communicate on issues, as this helps developers from around the world discuss, share experiences, and answer questions together.

For business licensing and other related inquiries, don't hesitate to contact `binyanrui@gmail.com`.

## 🔆 Introduction
🤗 If you find NormalCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

🔥 NormalCrafter can generate temporally consistent normal sequences
with fine-grained details from open-world videos with arbitrary lengths.

- `[24-04-01]` 🔥🔥🔥 **NormalCrafter** is released now, have fun!
## 🚀 Quick Start

### 🤖 Gradio Demo
- Online demo: [NormalCrafter](https://huggingface.co/spaces/Yanrui95/NormalCrafter) 
- Local demo:
    ```bash
    gradio app.py
    ``` 

### 🛠️ Installation
1. Clone this repo:
```bash
git clone git@github.com:Binyr/NormalCrafter.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```



### 🤗 Model Zoo
[NormalCrafter](https://huggingface.co/Yanrui95/NormalCrafter) is available in the Hugging Face Model Hub.

### 🏃‍♂️ Inference
#### 1. High-resolution inference, requires a GPU with ~20GB memory for 1024x576 resolution:
```bash
python run.py  --video-path examples/example_01.mp4
```

#### 2. Low-resolution inference requires a GPU with ~6GB memory for 512x256 resolution:
```bash
python run.py  --video-path examples/example_01.mp4 --max-res 512
```
