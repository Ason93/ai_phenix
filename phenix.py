'''
AI answer robot
Input: question
Output: answer audio

Process:
1.collect question audio(pyaudio)
2.question audio to text(baidu): denoise, ASR model(automatic speech recognition)
3.get answer text(turing robot)
4.answer text to audio(baidu)
5.play answer audio(pygame)

Bottleneck:
3.NLP model(natural language processing), dataset
'''

import os
import json
import time
import pygame
import pyaudio
import wave
import urllib.request
from tqdm import tqdm
from aip import AipSpeech


class Phenix(object):

    def __init__(self, config_path="config.json", log_dir="log"):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.config_path = config_path
        self.ques_audio_path = log_dir + "/question.wav"
        self.ans_audio_path = log_dir + "/answer.mp3"
        self.turing_url = "http://www.tuling123.com/openapi/api/v2"
        # https://console.bce.baidu.com/ai/#/ai/speech/app/detail~appId=3442079
        self.baidu_app_id = "27287198"
        self.baidu_api_key = "K19EvMBUjGVpFqv2enHZruyP"
        self.baidu_secret_key = "TYnAHhIEpjwaKz5HGHkudiPASadtSkOM"
        self.duration = 2  # ms
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.aipSpeech = AipSpeech(self.baidu_app_id, self.baidu_api_key,
                                   self.baidu_secret_key)

    def getQuetion(self):
        # 1.record question audio(pyaudio)
        print("Start recording question...")
        pa = pyaudio.PyAudio()
        stream = pa.open(format=self.format,
                         channels=self.channels,
                         rate=self.rate,
                         input=True,
                         frames_per_buffer=self.chunk)
        wf = wave.open(self.ques_audio_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pa.get_sample_size(self.format))
        wf.setframerate(self.rate)
        for i in tqdm(range(0, int(self.rate / self.chunk * self.duration))):
            data = stream.read(self.chunk)
            wf.writeframes(data)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        wf.close()
        print("End recording question...")

        # 2.question audio to text(baidu)
        def read_file(file_path):
            with open(file_path, 'rb') as fp:
                return fp.read()

        ques_context = self.aipSpeech.asr(read_file(self.ques_audio_path),
                                          'wav', self.rate, {
                                              'dev_pid': 1537,
                                              'lan': 'zh'
                                          })  # default 1537(mandarin)
        self.quest_text = str(ques_context['result'])
        print("Question: ", self.quest_text)

    def getAnswer(self):
        # 3.get answer text(turing robot)
        req = self._dump_json()
        http_post = urllib.request.Request(
            self.turing_url,
            data=req,
            headers={'content-type': 'application/json'})
        answer = urllib.request.urlopen(
            http_post)  # request url and return answer
        answer_str = answer.read().decode('utf8')
        answer_dict = json.loads(answer_str)  # answer string to dict
        intent_code = answer_dict.get('intent')['code']
        answer_text = answer_dict.get('results')[0]['values']['text']
        print("Answer: ", answer_text)
        # 4.answer text to audio(baidu)
        # zh: mandarin, 1: PC, vol: volume, spd: speed, pit: pitch, per: (0: female; 1: male; 2: unfettered; 4: Lolita)
        ans = self.aipSpeech.synthesis(answer_text, 'zh', 1, {
            'vol': 5,
            'per': 2,
            "spd": 5,
            "pit": 3,
            "per": 0
        })  # synthesize audio according to text
        # 5.play answer audio(pygame)
        if not isinstance(ans, dict):
            with open(self.ans_audio_path, 'wb') as f:
                f.write(ans)
            # time.sleep(1)
            pygame.mixer.init()
            print("Start playing answer ...")
            track = pygame.mixer.music.load(self.ans_audio_path)
            pygame.mixer.music.play()
            time.sleep(self.duration + 2)
            pygame.mixer.music.stop()
            pygame.quit()
            print("End playing answer ...")

    # for debug audio quality
    def playAudio(self):
        print("Start playing question...")
        wf = wave.open(self.ques_audio_path, 'rb')
        # instantiate PyAudio (1)
        p = pyaudio.PyAudio()
        # open stream (2)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        # read data
        data = wf.readframes(self.chunk)
        # play stream (3)
        datas = []
        while len(data) > 0:
            data = wf.readframes(self.chunk)
            datas.append(data)
        for d in tqdm(datas):
            stream.write(d)
        # stop stream (4)
        stream.stop_stream()
        stream.close()
        # close PyAudio (5)
        p.terminate()
        print("End playing question...")

    def _dump_json(self):
        with open(self.config_path, 'r', encoding='utf-8') as f_json:
            req = json.load(f_json)
        req['perception']['inputText']['text'] = self.quest_text
        req = json.dumps(
            req,
            sort_keys=True,
            indent=4,
        ).encode('utf8')
        return req


if __name__ == "__main__":
    phx = Phenix()
    phx.getQuetion()
    phx.getAnswer()
