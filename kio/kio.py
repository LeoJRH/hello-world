from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
#帮助理解特征值，分词等用法


# vector = CountVectorizer()
#
# res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])
#
# print(vector.get_feature_names())
#
# print(res.toarray())

def dictvec():
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{'city': 'Beijiing', 'temperature': 100},
                        {'city': 'Shanghai', 'temperature': 150},
                        {'city': 'Jiangxi', 'temperature': 50}])
    print(dict.get_feature_names() )
    print(dict.inverse_transform(data))

    # print(data)

def countdev():
    ki = jieba.cut("人生苦短,我喜欢PYTHON")


    kio=list(ki)
    k = ' '.join(kio)
    print(k)
    cv = CountVectorizer()
    datea= cv.fit_transform([k])
    print(cv.get_feature_names())
    print(datea.toarray())

    return None


def cutword():
    c1="开发者都或多或少接触过 linux 接触过命令行，当然肯定也都被命令行狠狠地“fuck”过。我很多时候都是微不足道的原因导致了命令行出错，例如将 python 输入成 ptyhon，例如将 ls -alh 输入成 ls a-lh而导致出错，这个时候我会想说：“fuck”。"
    c2="开发 thefuck 的这位同仁，恐怕也经常会有这种不和谐的情况。因此开发了这个软件 thefuck。thefuck 不仅仅能修复字符输入顺序的错误，在很多别的你想说“fuck”的情况下，thefuck 依然有效。例如以下情况。"
    c3="任何情况下你想说“我操”，你都可以用得到 thefuck。"

    content1 = jieba.cut(c1)
    content2 = jieba.cut(c2)
    content3 = jieba.cut(c3)

    co1 = ' '.join(content1)
    co2 = ' '.join(content2)
    co3 = ' '.join(content3)
    return co1,co2,co3

def tfidfvec():
    c1 ,c2 ,c3 = cutword()
    print(c1,c2,c3)
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1,c2,c3])
    print(tf.get_feature_names())
    print(data.toarray())


if __name__ == "__main__":
    tfidfvec()
    # dictvec()
