import telebot
from telebot import types
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pyvips
import time
import io

facetracker = load_model('C:/Users/Admin/Face Detection/facetracker_02.h5')
sextracker = load_model('C:/Users/Admin/Face Detection/sextracker.h5')


print('Starting up bot...')

TOKEN = '6240085199:AAFgxh4heqGjq5XCEBvj2obvX2l9eulNIcM'
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def handle_start_command(message):
    # Get the chat ID of the user who sent the message
    chat_id = message.chat.id
    
    # Send the response message
    bot.send_message(chat_id, 'Загрузите фото - нейросеть выделит лицо и оценит насколько оно секси. Или нет...')


@bot.message_handler(content_types=['photo'])
def handle_media(message):
    if message.content_type == 'photo':
        # Получаем информацию о фотографии
        photo = message.photo[-1]
        photo_id = photo.file_id
        
        # Получаем объект файла фотографии
        file_info = bot.get_file(photo_id)
        
        # Скачиваем файл фотографии
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Сохраняем файл фотографии на диск
        with open('photo.jpg', 'wb') as f:
            f.write(downloaded_file)
        
        # Читаем фотографию с помощью cv2.imread
        img = cv2.imread('photo.jpg')

        height, width, _ = img.shape

        if height < width:
            diff = width - height
            pad = diff // 2
            padded_img = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='constant', constant_values=128)
        else:
            diff = height - width
            pad = diff // 2
            padded_img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_values=128)




        

        
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (120, 120), interpolation=cv2.INTER_AREA)
        
        yhat = facetracker.predict(np.expand_dims(resized/255,0))
        sample_coords = yhat[1][0]

        print(yhat[0])
        
        if yhat[0] > 0.5: 

            padded_height, padded_width, _ = padded_img.shape

            # Controls the main rectangle
            top_left = np.multiply(sample_coords[:2], [padded_width, padded_height]).astype(int)
            bottom_right = np.multiply(sample_coords[2:], [padded_width, padded_height]).astype(int)

            #Предсказание красоты
            padded_img_sex = padded_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

            resize_sex = tf.image.resize(padded_img_sex, (120,120))

            yhat = sextracker.predict(np.expand_dims(resize_sex/255, 0))
            if yhat > 0.5: 
                sex = f'Ugly {"{:.0f}".format(100 * yhat[0][0])}%'
            else:
                sex = f'Sexy {"{:.0f}".format(100 - 100 * yhat[0][0])}%'


            print(top_left[1])
            print(pad)

            #Поправка если прямоугольник выходит за верхнюю границу (не будет видно надписи)
            if height < width:
                if top_left[1] < (pad):
                    top_left[1] = (pad)

            print(top_left[1])
            print(pad)
                  
        

            cv2.rectangle(padded_img, top_left, bottom_right, (255,0,0), 2)
            
            # Controls the label rectangle
            cv2.rectangle(padded_img, 
                        tuple(np.add(top_left, [0,30])),
                        tuple(np.add(top_left, [165,0])), 
                        (255,0,0), -1)
            
            # Controls the text rendered
            cv2.putText(padded_img, sex, tuple(np.add(top_left, [0,25])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if height < width:            
                padded_img = padded_img[pad:(padded_height-pad), :width, :]
            else:
                padded_img = padded_img[:height, pad:(padded_height-pad), :]
                    
            # Сохранение изображения с прямоугольником
            cv2.imwrite('output.jpg', padded_img)
            
            

            # Отправка изображения с прямоугольником
            with open('output.jpg', 'rb') as f:
                bot.send_photo(message.chat.id, f)


            os.remove('output.jpg')
            
        else:
            bot.send_message(message.chat.id, 'Бот не видит здесь лица.')
        
        



        os.remove('photo.jpg')


bot.polling()