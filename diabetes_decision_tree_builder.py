
# init accuracy, max loop, and loop count
_accuracy = 0.0
_maxLoop = 10000
_loopCount = 0
_timeUpdate = 0

# read csv using pandas library
import pandas as pd
df = pd.read_csv("data/diabetes.csv", header=0)
cols = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
# feature importace result per column using ExtraTreesClassifier
# [ 0.09954061  0.24259426  0.09109709  0.08101766  0.07893924  0.14250498 0.12221649  0.14208966]
# reduced columns based on feature selection == ["preg", "plas", "pres", "mass", "pedi", "age"]
df["class"] = df["class"].apply(lambda x: 0 if x == 'tested_negative' else 1)

while _loopCount < _maxLoop :
    # print every 1000 iteration
    if _loopCount % 1000 == 0 : 
        print ("Current loop count: {} \nNumber of time tree updated: {} \nBest accuracy: {} \n".format(_loopCount, _timeUpdate, _accuracy))

    # split test and train data
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size = 0.5)
    train_cols = train_df[list(cols)].values
    train_target = train_df["class"].values
    test_cols = test_df[list(cols)].values
    test_target = test_df["class"].values

    # build decision tree using training data
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf.fit(train_cols, train_target)

    # test decision tree using test data
    predict = clf.predict(test_cols)
    false_prediction = 0
    for index, val in enumerate (predict) :
        if test_target[index] != val :
            false_prediction = false_prediction + 1

    # calculate new accuracy
    _newAccuracy = float (len (train_cols) - false_prediction) / float (len (test_cols)) * 100

    # check if this tree is better than the previous tree
    if _newAccuracy > _accuracy :
        # renew accuracy and add update count
        _accuracy = _newAccuracy
        _timeUpdate = _timeUpdate + 1

        # save train data
        train_df.to_csv("data/diabetes_train.csv")
        # save_train = open ("data/jantung_train.csv", 'w')
        # save_train.write(train_cols)

        # save test data
        test_df.to_csv("data/diabetes_test.csv")
        # save_test = open ("data/jantung_test.csv", 'w')
        # save_test.write(test_df.values)

        # save tree
        from sklearn.externals.six import StringIO
        with open ("result/diabetes.dot", 'w') as f :
            f = tree.export_graphviz (clf, out_file=f, feature_names=cols)
        
        # export tree
        import os
        os.system('dot -Tpng result/diabetes.dot -o result/diabetes.png')
    
    # increment loop counter
    _loopCount = _loopCount + 1

print ("Final accuracy: {} \nDone.".format(_accuracy))