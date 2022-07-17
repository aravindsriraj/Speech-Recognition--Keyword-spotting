from gtts import gTTS

import os


mytext = 'right'

language = 'en'


myobj = gTTS(text=mytext, lang=language, slow=False)


myobj.save("right.wav")


