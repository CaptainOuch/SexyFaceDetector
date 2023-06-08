# Определние лица и его сексуальности с помощью сверточных нейронных сетей

# Результат:
Есть 2 категории

Sexy:
<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/38643187/244308863-25d92455-a100-4eec-b5fd-1a217433810a.png"/>
</p>

Ugly:
<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/38643187/244309258-c7aac061-dfcb-4e0e-ad87-da8de4f6f491.png"/>
</p>

## Данные

Данные
- Данные для определения координат лица были сняты с камеры с помощью
   "source": [
    "cap = cv2.VideoCapture(1)\n",
    "for imgnum in range(number_images):\n",
    "    print('Collecting image {}'.format(imgnum))\n",
    "    ret, frame = cap.read()\n",
    "    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')\n",
    "    cv2.imwrite(imgname, frame)\n",
    "    cv2.imshow('frame', frame)\n",
    "    time.sleep(0.5)\n",
    "\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        break\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
- [ImageNet 1K Dataset](http://image-net.org/download-images) (ensure it can be loaded by torchvision.datasets.Imagenet)

<p id="train_fer"></p>

## Training on FER2013

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IEQ091jBeJrOKHJe4wNhodH-bUGbLHSE?usp=sharing)

- To train network, you need to specify model name and other hyperparameters in config file (located at configs/\*) then ensure it is loaded in main file, then run training procedure by simply run main file, for example:

```Shell
python main_fer.py  # Example for fer2013_config.json file
```

- The best checkpoints will chosen at term of best validation accuracy, located at `saved/checkpoints`
- The TensorBoard training logs are located at `saved/logs`, to open it, use `tensorboard --logdir saved/logs/`

<p align="center">
<img width=900 src= "https://user-images.githubusercontent.com/24642166/75408653-fddf2b00-5948-11ea-981f-3d95478d5708.png"/>
</p>

- By default, it will train `alexnet` model, you can switch to another model by edit `configs/fer2013\_config.json` file (to `resnet18` or `cbam\_resnet50` or my network `resmasking\_dropout1`.

<p id="train_imagenet"></p>

## Training on Imagenet dataset

To perform training resnet34 on 4 V100 GPUs on a single machine:

```Shell
python ./main_imagenet.py -a resnet34 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

<p id="eval"></p>

## Evaluation

For student, who takes care of font family of confusion matrix and would like to write things in LaTeX, below is an example for generating a striking confusion matrix.

(Read [this article](https://matplotlib.org/3.1.1/tutorials/text/usetex.html) for more information, there will be some bugs if you blindly run the code without reading).

```Shell
python cm_cbam.py
```

<p align="center">
<img width=600 src= "https://user-images.githubusercontent.com/24642166/104806916-81c62e00-580d-11eb-8dcd-c5759e5d48ae.png"/>
</p>

## Ensemble method

I used no-weighted sum avarage ensemble method to fusing 7 different models together, to reproduce results, you need to do some steps:

1. Download all needed trained weights and located on `./saved/checkpoints/` directory. Link to download can be found on Benchmarking section.
2. Edit file `gen_results` and run it to generate result offline for **each** model.
3. Run `gen_ensemble.py` file to generate accuracy for example methods.

<p id="docs"></p>

## Dissertation and Slide

- [Dissertation PDF (in Vietnamese)](https://drive.google.com/drive/folders/1Nt7y1T99HpmF93peYxMg-i6BUqdzDBve?usp=sharing)
- [Dissertation Overleaf Source](https://www.overleaf.com/read/qdyhnzjmbscd)
- [Presentation slide PDF (in English) with full appendix](https://drive.google.com/drive/folders/1Nt7y1T99HpmF93peYxMg-i6BUqdzDBve?usp=sharing)
- [Presentation slide Overleaf Source](https://www.overleaf.com/read/vxdhjvhvgwdn)
- [Paper](docs/paper.pdf)

<p id="author"></p>

## Authors

- [**Luan Pham**](https://github.com/phamquiluan)
- [**Tuan Anh Tran**](https://github.com/phamquiluan)


<p id="references"></p>

## Citation

Pham, Luan, The Huynh Vu, and Tuan Anh Tran. "Facial expression recognition using residual masking network." 2020 25Th international conference on pattern recognition (ICPR). IEEE, 2021.
```
@inproceedings{pham2021facial,
  title={Facial expression recognition using residual masking network},
  author={Pham, Luan and Vu, The Huynh and Tran, Tuan Anh},
  booktitle={2020 25Th international conference on pattern recognition (ICPR)},
  pages={4513--4519},
  year={2021},
  organization={IEEE}
}
```
