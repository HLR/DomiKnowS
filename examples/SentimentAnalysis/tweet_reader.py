

from examples.SentimentAnalysis.reader_to_jason import SentimentReader

a = SentimentReader("twitter_data/train5k.csv", "csv")

for l in a.run():
    print(l)
#
# def myReader():
#  filename = "twitter_data/train5k.csv"
#  with open(filename, encoding='latin-1') as f:
#      content = f.readlines()
#  return content
#
# for line in myReader():
#  print(line)