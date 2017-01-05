# [ID] Generate Decision Tree menggunakan library Scikit-Learn, Pandas dan Numpy
# [ENG] Generate Decision Tree using Scikit-Learn, Pandas and Numpy

# function untuk membangun decision tree
def treeBuildAndTest (jenis, train_feature, train_target, test_feature, test_target, cols, output_file, output_png):
    # hitung jumlah test data
    jumlahTestData = len(test_target)

    # import tree dari sklearn untuk membangun decision tree
    from sklearn import tree

    # generate decision tree dengan kriteria percabangan dengan Entropy
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(train_feature, train_target)
    
    # generate .dot file untuk di convert menjadi .png
    # import StringIO dari sklearn.external.six untuk membantu generate .dot file
    from sklearn.externals.six import StringIO
    with open(output_file, 'w') as f:
        f = tree.export_graphviz(clf, out_file=f, feature_names=cols)
    print ("Lokasi output tree: {}".format(output_file))

    # convert .dot menjadi .png
    import os
    os.system('dot -Tpng {} -o {}'.format(output_file, output_png))
    print ("Lokasi output gambar: {}".format(output_png))

    # test tree yang telah dibangun dengan test data yang sudah ditentukan
    prediksi = clf.predict(test_feature)

    # hitung akurasi decision tree
    prediksiSalah = 0
    for index, val in enumerate(prediksi):
        if test_target[index] != val:
            prediksiSalah = prediksiSalah + 1
    
    # perhitungan
    jumlahPrediksiBenar = jumlahTestData-prediksiSalah
    totalUlang = jumlahPrediksiBenar+prediksiSalah
    totalakurasi = float(jumlahPrediksiBenar) / float(jumlahTestData) * 100

    # debug
    # print ("Benar: {}".format(jumlahPrediksiBenar))
    # print ("Salah: {}".format(prediksiSalah))
    # print ("Total: {}".format(totalUlang))
    # print ("Akurasi: %.2f %%"%(totalakurasi))

    # tampilkan hasil perhitungan dan akurasi
    prediksiBenar = 'Jumlah prediksi yang benar: {} dari {} jawaban'.format(jumlahPrediksiBenar, jumlahTestData)
    akurasi = 'Akurasi: %.2f %%'%(totalakurasi)
    resultBuilder = prediksiBenar + "\n" + akurasi + "\n"
    
    # end function
    return resultBuilder

# mulai program
# import pandas untuk membaca data dari csv
import pandas as pd

# lokasi data
lokasiData_jantung = "data/jantung.csv"
lokasiData_diabetes = "data/diabetes.csv"

# header data jantung & diabetes ada di baris pertama, 
# separator di file jantung.csv pakai ';'
# separator di file diabetes.csv pakai ','
jantung_df = pd.read_csv(lokasiData_jantung, header=0, sep=";")
diabetes_df = pd.read_csv(lokasiData_diabetes, header=0)

# pada data diabetes, kolom class masih berupa kalimat, ubah menjadi angka
# untuk tested_negative bernilai 0 dan tested_positive bernilai 1
diabetes_df["class"] = diabetes_df["class"].apply(lambda x: 0 if x == 'tested_negative' else 1)

# pisah data menjadi data training dan data testing
# jumlah data training = 2/3 dari total data (0.6666)
# jumlah data testing = 1/3 dari total data (0.3333)
# import sklearn.model_selection.train_test_split untuk membantu pemisahan data
from sklearn.model_selection import train_test_split
# data jantung
jantung_train, jantung_test = train_test_split(jantung_df, test_size = 0.3)
# data diabetes
diabetes_train, diabetes_test = train_test_split(diabetes_df, test_size = 0.3)


# menentukan kolom fitur yang akan digunakan untuk membangun tree
# fitur jantung
jantung_columns = ["age", "cp", "trestbp", "fbs", "restecg", "thalach"]
# jantung_columns = ["age", "cp", "trestbp", "thalach"]
# hasil feature importace menggunakan ExtraTreesClassifier
# [ 0.26346508  0.13608912  0.2459563   0.02467597  0.03230045  0.29751308]
jantung_train_features = jantung_train[list(jantung_columns)].values
jantung_train_target = jantung_train["num"].values
jantung_test_features = jantung_test[list(jantung_columns)].values
jantung_test_target = jantung_test["num"].values
#  fitur diabetes
diabetes_columns = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
# diabetes_columns = ["preg", "plas", "pres", "mass", "pedi", "age"]
# hasil feature importace menggunakan ExtraTreesClassifier
# [ 0.09954061  0.24259426  0.09109709  0.08101766  0.07893924  0.14250498 0.12221649  0.14208966]
diabetes_train_features = diabetes_train[list(diabetes_columns)].values
diabetes_train_target = diabetes_train["class"].values
diabetes_test_features = diabetes_test[list(diabetes_columns)].values
diabetes_test_target = diabetes_test["class"].values


# generate decision tree dan output berupa .dot file
# convert .dot file menjadi .png dengan command: dot -Tpng <file>.dot -o <file>.png
print ("\nMemproses data jantung...\nSource data jantung: {}".format(lokasiData_jantung))
proses_jantung = treeBuildAndTest("jantung", jantung_train_features, jantung_train_target, jantung_test_features, jantung_test_target, jantung_columns, "result/jantung.dot", "result/jantung.png")
print("\nMemproses data diabetes...\nSource data diabetes: {}".format(lokasiData_diabetes))
proses_diabetes = treeBuildAndTest("diabetes", diabetes_train_features, diabetes_train_target, diabetes_test_features, diabetes_test_target, diabetes_columns, "result/diabetes.dot", "result/diabetes.png")
print("Proses Selesai\n")

# print hasil
print ("Hasil proses jantung: \n"+proses_jantung)
print ("Hasil proses diabetes: \n"+proses_diabetes)
print ("END\n")