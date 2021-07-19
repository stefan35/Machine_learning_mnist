import numpy
import pandas
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import zero_one_loss
from sklearn.svm import SVR
from mlens.ensemble import BlendEnsemble
from mlens.metrics.metrics import rmse

def dec_tree():
    # Vo funkcii nacitam do premennych trenovaci a testovaci dataset zo suboru
    # Nasledne z tychto premennych osobitne nacitam label a samostatne data
    # Data nacitavam do dvoch roznych premennych lebo do jednej budem appendovat atributy
    print("--- Decision Tree ---")
    print("Loading training input... ")
    train = pandas.read_csv("train_10000.csv")
    test = pandas.read_csv("test_10000.csv")

    image_attributes = []
    image = []
    test_label = []
    for row in train.iterrows():
        label_number = row[1][0]
        image_number_attributes = numpy.array(row[1][1:]) / 255
        image_number = numpy.array(row[1][1:]) / 255
        blank_space = numpy.count_nonzero(image_number_attributes == 0.0)
        number_space = ((28 * 28) - blank_space)
        number_space = numpy.array(number_space)
        bottom_r = numpy.array([image_number_attributes[581]])
        bottom_m = numpy.array([image_number_attributes[574]])
        bottom_l = numpy.array([image_number_attributes[567]])
        mid_r = numpy.array([image_number_attributes[385]])
        mid_m = numpy.array([image_number_attributes[378]])
        top_m = numpy.array([image_number_attributes[182]])
        image_number_attributes = numpy.append(image_number_attributes, number_space)
        image_number_attributes = numpy.append(image_number_attributes, bottom_r)
        image_number_attributes = numpy.append(image_number_attributes, bottom_m)
        image_number_attributes = numpy.append(image_number_attributes, bottom_l)
        image_number_attributes = numpy.append(image_number_attributes, mid_r)
        image_number_attributes = numpy.append(image_number_attributes, mid_m)
        image_number_attributes = numpy.append(image_number_attributes, top_m)
        image_attributes.append(image_number_attributes)
        image.append(image_number)
        test_label.append(label_number)
    image_attributes = numpy.array(image_attributes)
    image = numpy.array(image)
    test_label = numpy.array(test_label)

    #Trenovanie modelu na datach bez atributov a aj s atributmi
    print("Training...")
    dcst = tree.DecisionTreeClassifier(criterion="entropy")
    dcst = dcst.fit(image, test_label)
    dcst_attributes = tree.DecisionTreeClassifier(criterion="entropy")
    dcst_attributes = dcst_attributes.fit(image_attributes, test_label)

    #Nacitanie testovacieho datasetu
    #Nacitavam to tak isto ako treningovy dataset
    #Pridavam tie iste data na tie iste miesta ako pri nacitani treningoveho datasetu
    print("Loading testing input...")
    test_attributes = []
    test_input = []
    test_label = []
    for row in test.iterrows():
        label_number = row[1][0]
        image_number_attributes = numpy.array(row[1][1:]) / 255
        image_number = numpy.array(row[1][1:]) / 255
        blank_space = numpy.count_nonzero(image_number_attributes == 0.0)
        number_space = ((28 * 28) - blank_space)
        number_space = numpy.array(number_space)
        bottom_r = numpy.array([image_number_attributes[581]])
        bottom_m = numpy.array([image_number_attributes[574]])
        bottom_l = numpy.array([image_number_attributes[567]])
        mid_r = numpy.array([image_number_attributes[385]])
        mid_m = numpy.array([image_number_attributes[378]])
        top_m = numpy.array([image_number_attributes[182]])
        image_number_attributes = numpy.append(image_number_attributes, number_space)
        image_number_attributes = numpy.append(image_number_attributes, bottom_r)
        image_number_attributes = numpy.append(image_number_attributes, bottom_m)
        image_number_attributes = numpy.append(image_number_attributes, bottom_l)
        image_number_attributes = numpy.append(image_number_attributes, mid_r)
        image_number_attributes = numpy.append(image_number_attributes, mid_m)
        image_number_attributes = numpy.append(image_number_attributes, top_m)
        test_attributes.append(image_number_attributes)
        test_input.append(image_number)
        test_label.append(label_number)
    test_attributes = numpy.array(test_attributes)
    test_input = numpy.array(test_input)
    test_label = numpy.array(test_label)

    #Testovanie modelu na datach bez artibutov a aj s atributmi
    print("Testing...")
    pred_attributes = dcst_attributes.predict(test_attributes)
    predic = dcst.predict(test_input)

    #Vyhodnotenie, vypisanie uspesnosti, error rate a aj matrix
    #Prv sa vypise vyhodnotenie pre model s atributmi a hned za nim pre model bez atributov
    print()
    print("Result with attributes...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, pred_attributes)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, pred_attributes)) * 100, 2), "%")
    print("Matrix")
    print(confusion_matrix(test_label, pred_attributes))

    print()
    print("Result without attributes...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, predic)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, predic)) * 100, 2), "%")
    print("Matrix")
    print(confusion_matrix(test_label, predic))

def backprop():
    #Nacitanie datasetu pre trening a pre testovanie zo suboru
    #Najprv sa nacita treningovy dataset v ktorom si osobitne nacitam label a data
    #Data nacitam do dvoch premennych do druhej appendujem atributy
    print("--- Backpropagation ---")
    print("Loading training input... ")
    train = pandas.read_csv("train_10000.csv")
    test = pandas.read_csv("test_10000.csv")

    image_attributes = []
    image = []
    test_label = []
    for row in train.iterrows():
        label_number = row[1][0]
        image_number_attributes = numpy.array(row[1][1:]) / 255
        image_number = numpy.array(row[1][1:]) / 255
        blank_space = numpy.count_nonzero(image_number_attributes == 0.0)
        number_space = ((28 * 28) - blank_space)
        bottom_m = numpy.array([image_number_attributes[574]])
        mid_m = numpy.array([image_number_attributes[378]])
        top_m = numpy.array([image_number_attributes[182]])
        image_number_attributes = numpy.append(image_number_attributes, (blank_space / number_space))
        image_number_attributes = numpy.append(image_number_attributes, bottom_m)
        image_number_attributes = numpy.append(image_number_attributes, mid_m)
        image_number_attributes = numpy.append(image_number_attributes, top_m)
        image_attributes.append(image_number_attributes)
        image.append(image_number)
        test_label.append(label_number)
    image_attributes = numpy.array(image_attributes)
    image = numpy.array(image)
    test_label = numpy.array(test_label)

    #Trenovanie modelu na datach bez a aj s atributmi
    print("Training...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 75), max_iter=500, learning_rate="invscaling")
    mlp = mlp.fit(image, test_label)
    mlp_attributes = MLPClassifier(hidden_layer_sizes=(100, 50, 75), max_iter=500, learning_rate="invscaling")
    mlp_attributes = mlp_attributes.fit(image_attributes, test_label)

    #Take iste nacitanie testovacich dat ake bolo pri treningovych datach
    #Pridavam tie iste atributy na tie iste miesta ako pri treningovom datasete
    print("Loading testing input...")
    test_attributes = []
    test_input = []
    test_label = []
    for row in test.iterrows():
        label_number = row[1][0]
        image_number_attributes = numpy.array(row[1][1:]) / 255
        image_number = numpy.array(row[1][1:]) / 255
        blank_space = numpy.count_nonzero(image_number_attributes == 0.0)
        number_space = ((28 * 28) - blank_space)
        bottom_m = numpy.array([image_number_attributes[574]])
        mid_m = numpy.array([image_number_attributes[379]])
        top_m = numpy.array([image_number_attributes[182]])
        image_number_attributes = numpy.append(image_number_attributes, (blank_space / number_space))
        image_number_attributes = numpy.append(image_number_attributes, bottom_m)
        image_number_attributes = numpy.append(image_number_attributes, mid_m)
        image_number_attributes = numpy.append(image_number_attributes, top_m)
        test_attributes.append(image_number_attributes)
        test_input.append(image_number)
        test_label.append(label_number)
    test_attributes = numpy.array(test_attributes)
    test_input = numpy.array(test_input)
    test_label = numpy.array(test_label)

    #Testovanie modelu na testovacich datach s atributmi a bez atributov
    print("Testing...")
    pred_attributes = mlp_attributes.predict(test_attributes)
    predic = mlp.predict(test_input)

    #Vyhodnotenie, najpr sa vyhodnoti model s atributmi a potom bez atributov
    print()
    print("Result with attributes...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, pred_attributes)) * 100, 2))
    print("Error rate ", round((zero_one_loss(test_label, pred_attributes)) * 100, 2))
    print("Matrix")
    print(confusion_matrix(test_label, pred_attributes))

    print()
    print("Result without attributes...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, predic)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, predic)) * 100, 2), "%")
    print("Matrix")
    print(confusion_matrix(test_label, predic))

def rforest():
    #Nacitanie treningoveho a testovacieho datasetu zo suboru
    #Rozdelenie nacitanych dat na label a samotne data
    #Data sa nacitaju do dvoch premennych, do jednej tejto premennej sa budu appendovat atributy
    print("--- Random Forest ---")
    print("Loading training input... ")
    train = pandas.read_csv("train_10000.csv")
    test = pandas.read_csv("test_10000.csv")

    image_attributes = []
    image = []
    test_label = []
    for row in train.iterrows():
        label_number = row[1][0]
        image_number_attributes = numpy.array(row[1][1:]) / 255
        image_number = numpy.array(row[1][1:]) / 255
        blank_space = numpy.count_nonzero(image_number_attributes == 0.0)
        number_space = ((28 * 28) - blank_space)
        blank_space = numpy.array(blank_space)
        number_space = numpy.array(number_space)
        bottom_r = numpy.array([image_number_attributes[581]])
        mid_m = numpy.array([image_number_attributes[378]])
        top_l = numpy.array([image_number_attributes[175]])
        image_number_attributes = numpy.append(image_number_attributes, number_space)
        image_number_attributes = numpy.append(image_number_attributes, (blank_space / number_space))
        image_number_attributes = numpy.append(image_number_attributes, bottom_r)
        image_number_attributes = numpy.append(image_number_attributes, mid_m)
        image_number_attributes = numpy.append(image_number_attributes, top_l)
        image_attributes.append(image_number_attributes)
        image.append(image_number)
        test_label.append(label_number)
    image_attributes = numpy.array(image_attributes)
    image = numpy.array(image)
    test_label = numpy.array(test_label)

    #Trenovanie modelu na nacitanom treningovom datasete
    print("Training...")
    rfn_attributes = RandomForestClassifier(n_estimators=100)
    rfn = RandomForestClassifier(n_estimators=100)
    rfn_attributes.fit(image_attributes, test_label)
    rfn.fit(image, test_label)

    #Nacitanie testovacieho datasetu je take iste ako nacitanie treningoveho datasetu
    #Pridavam tie iste atributy na tie iste miesta ako pri treningovom datasete
    print("Loading testing input...")
    test_attributes = []
    test_input = []
    test_label = []
    for row in test.iterrows():
        label_number = row[1][0]
        image_number_attributes = numpy.array(row[1][1:]) / 255
        image_number = numpy.array(row[1][1:]) / 255
        blank_space = numpy.count_nonzero(image_number_attributes == 0.0)
        number_space = ((28 * 28) - blank_space)
        blank_space = numpy.array(blank_space)
        number_space = numpy.array(number_space)
        bottom_r = numpy.array([image_number_attributes[581]])
        mid_m = numpy.array([image_number_attributes[378]])
        top_l = numpy.array([image_number_attributes[175]])
        image_number_attributes = numpy.append(image_number_attributes, number_space)
        image_number_attributes = numpy.append(image_number_attributes, (blank_space / number_space))
        image_number_attributes = numpy.append(image_number_attributes, bottom_r)
        image_number_attributes = numpy.append(image_number_attributes, mid_m)
        image_number_attributes = numpy.append(image_number_attributes, top_l)
        test_attributes.append(image_number_attributes)
        test_input.append(image_number)
        test_label.append(label_number)
    test_attributes = numpy.array(test_attributes)
    test_input = numpy.array(test_input)
    test_label = numpy.array(test_label)

    #Testovanie modelu na testovacich datach
    print("Testing...")
    pred_attributes = rfn_attributes.predict(test_attributes)
    predic = rfn.predict(test_input)

    #Vyhodnotenie modelu s atributmi a hned za nim sa vyhodnoti model bez atributov
    print()
    print("Result with attributes...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, pred_attributes)) * 100, 2))
    print("Error rate ", round((zero_one_loss(test_label, pred_attributes)) * 100, 2))
    print("Matrix")
    print(confusion_matrix(test_label, pred_attributes))

    print()
    print("Result without attributes...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, predic)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, predic)) * 100, 2), "%")
    print("Matrix")
    print(confusion_matrix(test_label, predic))

def combination():
    #Nacitanie trenovacieho a testovacieho datasetu zo suboru
    #Pri nacitanie dat si nacitam label a potom hned za nim data
    print("--- Model Combination --")
    print("Loading input... ")
    train = pandas.read_csv("train_10000.csv")
    test = pandas.read_csv("test_10000.csv")

    image = []
    train_label = []
    for row in train.iterrows():
        label_number = row[1][0]
        image_number = numpy.array(row[1][1:]) / 255
        image.append(image_number)
        train_label.append(label_number)
    image = numpy.array(image)
    train_label = numpy.array(train_label)

    #Trenovanie troch modelov na trenovacom datasete
    print("Training...")
    dcst = tree.DecisionTreeClassifier(criterion="entropy")
    dcst = dcst.fit(image, train_label)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 75), max_iter=500, learning_rate="invscaling")
    mlp = mlp.fit(image, train_label)
    rfn = RandomForestClassifier(n_estimators=100)
    rfn.fit(image, train_label)

    #Nacitanie testovacieho datasetu
    #Nacitavam osobitne label a data
    print("Loading test input...")
    test_input = []
    test_label = []
    for row in test.iterrows():
        label_number = row[1][0]
        image_number = numpy.array(row[1][1:]) / 255
        test_input.append(image_number)
        test_label.append(label_number)
    test_input = numpy.array(test_input)
    test_label = numpy.array(test_label)

    #Testovanie modelov na testovacich datach
    print("Testing...")
    dcst_predic = dcst.predict(test_input)
    rfn_predic = rfn.predict(test_input)
    mlp_predic = mlp.predict(test_input)

    #Vyhodnotenie vsetkych troch modelov
    print("Decision Tree result...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, dcst_predic)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, dcst_predic)) * 100, 2), "%")

    print("Backpropagation result...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, mlp_predic)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, mlp_predic)) * 100, 2), "%")

    print("Random Forest result...")
    print("Accuracy ", round((metrics.accuracy_score(test_label, rfn_predic)) * 100, 2), "%")
    print("Error rate ", round((zero_one_loss(test_label, rfn_predic)) * 100, 2), "%")

    #Spojenie troch modelov do jedneho
    #Trenovanie a nasledne testovanie spojenych modelov po ktorom sa vypise uspesnot a error rate
    print("Merging...")
    ensemble = BlendEnsemble([RandomForestClassifier(n_estimators=100), tree.DecisionTreeClassifier(), MLPClassifier(hidden_layer_sizes=(100, 50, 75), max_iter=500, learning_rate="invscaling")], verbose=2)
    ensemble.add_meta(SVR(gamma='auto'))
    ensemble.fit(image, train_label)
    pred = ensemble.predict(test_input)
    print("All in one result...")
    print("Accuracy ", round(100 - rmse(test_label, pred), 2), "%")
    print("Error rate ", round(rmse(test_label, pred), 2), "%")

if __name__ == "__main__":
    print("1 - Decision tree")
    print("2 - Bakcpropagation")
    print("3 - Random Forest")
    print("4 - Combination of models")
    nb = input('Choose model number: ')
    number = int(nb)
    if number == 1:
        dec_tree()
    elif number == 2:
        backprop()
    elif number == 3:
        rforest()
    elif number == 4:
        combination()

