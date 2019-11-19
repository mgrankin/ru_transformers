from tendo import singleton
me = singleton.SingleInstance()

import logging

logging.basicConfig(filename="stihbot.log", level=logging.INFO)
logger = logging.getLogger(__name__)

import requests
url = 'https://models.dobro.ai/gpt2_poetry'

def get_sample(text):
    response = requests.post(url, data={"prompt": text, "length": 150})
    return response["replies"][0]

import json
data = json.load(open('config.json'))

from tendo import singleton
me = singleton.SingleInstance()

import telebot

bot = telebot.TeleBot(data['bot_key'])

from telebot import apihelper

apihelper.proxy = {'https':data['proxy_str']}

def message_handler(message):
    logger.info(message.from_user)
    with lock:
        try:
            bot.reply_to(message, get_sample(message.text))
        except telebot.apihelper.ApiException as e:
            print(e)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.reply_to(message, "Присылай начало, а я продолжу")

@bot.message_handler(func=lambda m: True)
def echo_all(message):
    message_handler(message)

@bot.channel_post_handler(func=lambda m: True)
def echo_all(message):
    message_handler(message)

bot.polling()


