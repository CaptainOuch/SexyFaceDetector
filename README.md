# Определение лица и его сексуальности с помощью сверточных нейронных сетей

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

# Данные
## Sexyness classifier

Данные для классификации сексуальности были загружены с помощью плагина для браузера 
- [Download All Images](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm) (Загружает изображения с любой страницы)
- В данном случае изображения были загружены по запросам в "google картинки" sexy face, ugly face

<p id="train_fer"></p>

## Face Detection
- Данные для определения координат лица были сняты с камеры с помощью функции .VideoCapture библиотеки opencv
```python
cap = cv2.VideoCapture(1)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

А затем размечены с помощью labelme

```python
!labelme
```

Далее данные были размножены с помощью библиотеки [Albumentations](https://albumentations.ai/)

```python
import albumentations as alb
```


```python
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
```
# Обучение моделей
## Sexyness classifier
В обучении классификатора использовался несложный пайплайн сверточной нейронной сети

```python
model.summary()
```

    Model: "sequential_14"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_23 (Conv2D)          (None, 118, 118, 16)      448       
                                                                     
     max_pooling2d_22 (MaxPoolin  (None, 59, 59, 16)       0         
     g2D)                                                            
                                                                     
     conv2d_24 (Conv2D)          (None, 57, 57, 32)        4640      
                                                                     
     max_pooling2d_23 (MaxPoolin  (None, 28, 28, 32)       0         
     g2D)                                                            
                                                                     
     conv2d_25 (Conv2D)          (None, 26, 26, 64)        18496     
                                                                     
     max_pooling2d_24 (MaxPoolin  (None, 13, 13, 64)       0         
     g2D)                                                            
                                                                     
     flatten_7 (Flatten)         (None, 10816)             0         
                                                                     
     dense_21 (Dense)            (None, 512)               5538304   
                                                                     
     dense_22 (Dense)            (None, 1)                 513       
                                                                     
    =================================================================
    Total params: 5,562,401
    Trainable params: 5,562,401
    Non-trainable params: 0
    _________________________________________________________________


<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/38643187/244346074-8f51af47-1a58-4d6f-843a-da0630820d77.png"/>
</p>
<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/38643187/244346192-cf8fb7f6-de7f-47c9-80f5-1266f234a1e0.png"/>
</p>


## Face Detection
В пайплайне модели использовалась предобученная сверточная нейронная сеть VGG16 без конечного слоя с нейронами

```python
vgg = VGG16(include_top=False)
```
<p align="center">
<img width=500 src= "https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network-1.jpg"/>
</p>

```python
facetracker.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 120, 120, 3  0           []                               
                                    )]                                                                
                                                                                                      
     vgg16 (Functional)             (None, None, None,   14714688    ['input_2[0][0]']                
                                    512)                                                              
                                                                                                      
     global_max_pooling2d (GlobalMa  (None, 512)         0           ['vgg16[0][0]']                  
     xPooling2D)                                                                                      
                                                                                                      
     global_max_pooling2d_1 (Global  (None, 512)         0           ['vgg16[0][0]']                  
     MaxPooling2D)                                                                                    
                                                                                                      
     dense (Dense)                  (None, 2048)         1050624     ['global_max_pooling2d[0][0]']   
                                                                                                      
     dense_2 (Dense)                (None, 2048)         1050624     ['global_max_pooling2d_1[0][0]'] 
                                                                                                      
     dense_1 (Dense)                (None, 1)            2049        ['dense[0][0]']                  
                                                                                                      
     dense_3 (Dense)                (None, 4)            8196        ['dense_2[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 16,826,181
    Trainable params: 16,826,181
    Non-trainable params: 0
    __________________________________________________________________________________________________

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
