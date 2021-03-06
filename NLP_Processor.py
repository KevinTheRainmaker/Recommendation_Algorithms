import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class TextProcessor:
    """
    Class for carrying all the text pre-processing stuff throughout the project
    """

    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.ps = PorterStemmer()

        # stemmer will be used for each unique word once
        self.stemmed = dict()

    def process(self, text: str, allow_stopwords: bool = False) -> str:
        """
        Process the specified text,
        splitting by non-alphabetic symbols, casting to lower case,
        removing stopwords, HTML tags and stemming each word

        :param text: text to precess
        :param allow_stopwords: whether to remove stopwords
        :return: processed text
        """
        ret = []

        # split and cast to lower case
        text = re.sub(r'<[^>]+>', ' ', str(text))
        for word in re.split('[^a-zA-Z]', str(text).lower()):
            # remove non-alphabetic and stop words
            if (word.isalpha() and word not in self.stopwords) or allow_stopwords:
                if word not in self.stemmed:
                    self.stemmed[word] = self.ps.stem(word)
                # use stemmed version of word
                ret.append(self.stemmed[word])
        return ' '.join(ret)

# print(TextProcessor.process.__annotations__)

tp = TextProcessor()
