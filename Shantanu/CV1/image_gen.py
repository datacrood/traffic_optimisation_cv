# -*- coding: utf-8 -*-

import subprocess
import os
import json
import codecs

KANJI = "hirigana_numbers_katakana.txt"
with open('data/fonts-list.json') as f:
    FONTS = json.load(f)

for FONT_SHORT, FONT_FULL in FONTS.iteritems():
    directory = "data/kanji-%s" % FONT_SHORT

    if not os.path.exists(directory):
      os.makedirs(directory)

    with codecs.open(KANJI, encoding='utf-8') as f:
      idx = 2136
      for l in f:
        text = l.strip()
        if not text or len(text) == 0: continue
        idx += 1
        cmd = u"""
          convert -font "%s" -size 256x -background white -fill black label:"%s" data/kanji-%s/kanji_%s.png
        """ % (unicode(FONT_FULL), unicode(text), unicode(FONT_SHORT), unicode(str(idx)))
        print cmd
        subprocess.call(cmd, shell=True)
