import rus_pos
import nltk

tagger = rus_pos.get_tagger()
m = 'Эти типы стали есть на складе'
nm = list(map(nltk.word_tokenize, nltk.sent_tokenize(m)))
print(tagger.pos_tag(nm))
