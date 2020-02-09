from tendo import singleton
me = singleton.SingleInstance()

import logging

logging.basicConfig(filename="stihbot.log", level=logging.INFO)
logger = logging.getLogger(__name__)

import json
data = json.load(open('config.json'))

import requests
url = data['url']
length = data['length']

def get_sample(text):
    if 'poetry' in url:
        response = requests.post(url, json={"prompt": text, "length": length})
    else:
        response = requests.post(url, json={"prompt": text, "length": length, "num_samples": 1, "allow_linebreak": False})

    print(response)
    return json.loads(response.text)["replies"][0]

import telebot

bot = telebot.TeleBot(data['bot_key'], num_threads=20)

from telebot import apihelper

#apihelper.proxy = {'https':data['proxy_str']}

def message_handler(message):
    logger.info(message.from_user)
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        bot.reply_to(message, f'__{message.text}__' + get_sample(message.text), parse_mode="Markdown")
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


