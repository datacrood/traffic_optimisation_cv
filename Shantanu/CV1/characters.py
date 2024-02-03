import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import codecs

FONT_DIR_BASE = "data/kanji-"
FONTS = ('Mincho', 'MPlusThin', 'HonyaJi')
FONTS_DIRS = [FONT_DIR_BASE + f for f in FONTS]

with codecs.open('kanji_list.txt', encoding='utf-8') as f:
  FONT_LIST = [l.strip() for l in f.readlines()]

class Character(object):

  def __init__(self, idnum):
    self.idnum = idnum

  def show(self):
    fig = plt.figure()
    for i, d in enumerate(FONTS_DIRS):
      sub = fig.add_subplot(1, len(FONTS_DIRS), i+1)
      im = mpimg.imread("%s/kanji_%s.png" % (d, self.idnum))
      plt.imshow(im)
    plt.show()

  def get_str(self):
    return unicode(FONT_LIST[self.idnum])

  def from_str(ch):
    for i, ch1 in enumerate(FONT_LIST):
        if ch == ch1:
            return Character(i+1)
    return None

def main():
  if len(sys.argv) != 2:
    print 'Pass id as only arg'
    return
  ch = Character(int(sys.argv[1]))
  print u'Displaying %s' % ch.get_str()
  ch.show()

if __name__ == "__main__":
  main()
