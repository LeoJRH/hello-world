from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
li = load_iris()


# archive_path = r'/home/leo/scikit_learn_data/20news_home/20news-bydate.tar.gz'

# print("Get TZ:")
# print(li.data)
# print("Get MB:")
# print((li.target))
# print(li.DESCR)
# Return both train x_train,ytrain and test x_test y_test
#x ---TZ    y ---XL
# x_train,x_test,y_train,y_test = train_test_split(li.data, li.target, test_size=0.25)
#
# print("Train x and y:", x_train, y_train)
# print("Test x and y:", x_test, y_test)

# news = fetch_20newsgroups(subset='all')
# print(news.data)
# print(news.target)

def naviebayes():
    news = fetch_20newsgroups(subset='all')

    #Data split
    x_train,x_test,y_train,y_test = train_test_split(news.data, news.target,test_size=0.25)

    # feature extraction
    tf = TfidfVectorizer()
    #Use x_train to make statistics of each arctics 's importance
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    x_test = tf.transform(x_test)
    # bATES calculate

    mlt = MultinomialNB(alpha=1.0)
    print(x_train)
    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)
    #Score
    print("Predict kind is", y_predict)
    print("The score is ",mlt.score(x_test,y_test))

    return None

if __name__ == '__main__':
    naviebayes()