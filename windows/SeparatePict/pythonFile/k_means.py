def near(vector, center_vectors):
    """
    vectorに対し、尤も近いラベルを返す
    :param vector:
    :param center_vectors:
    :return:
    """
    euclid_dist = lambda vector1, vector2: (sum([(vec[0]-vec[1])**2 for vec in list(zip(vector1, vector2))]))**0.5
    d = [euclid_dist(vector, center_vector) for center_vector in center_vectors]
    return d.index(min(d))

def clustering_class1(vector , label_count, max1, min1, learning_count_max=1000):
    """
    K-meansを行い、各ラベルの重心を返す
    :param vectors:
    :param label_count:
    :param learning_count_max:
    :return:
    """
    import random
    #各vectorに割り当てられたクラスタラベルを保持するvector
    label_vector = [random.randint(0, label_count-1) for i in vectors]
    #一つ前のStepで割り当てられたラベル。終了条件の判定に使用
    old_label_vector = list()
    #各クラスタの重心vector
    center_vectors = [[0 for i in range(len(vectors[0]))] for label in range(label_count)]

    for step in range(learning_count_max):
        #各クラスタの重心vectorの作成
        if (label_count==2)and(step==0):
            center_vectors[0] = [min1[0], min1[1]]
            center_vectors[1] = [max1[0], max1[1]]
        if (label_count==3)and(step==0):
            center_vectors[0] = [min1[0], min1[1]]
            center_vectors[1] = [(min1[0]+max1[0])/4, (min1[1]+max1[1])/4]
            center_vectors[2] = [max1[0], max1[1]]
        #各ベクトルのラベルの再割当て
        for i, vec in enumerate(vectors):
            label_vector[i] = near(vec, center_vectors)
        #前Stepと比較し、ラベルの割り当てに変化が無かったら終了
        if old_label_vector == label_vector:
            break
        #ラベルのベクトルを保持
        old_label_vector = [l for l in label_vector]
    return center_vectors

def clustering_class2(vectors, label_count, max1, min1, learning_count_max=1000):
    """
    K-meansを行い、各ラベルの重心を返す
    :param vectors:
    :param label_count:
    :param learning_count_max:
    :return:
    """
    import random
    #各vectorに割り当てられたクラスタラベルを保持するvector
    label_vector = [random.randint(0, label_count-1) for i in vectors]
    #一つ前のStepで割り当てられたラベル。終了条件の判定に使用
    old_label_vector = list()
    #各クラスタの重心vector
    center_vectors = [[0 for i in range(len(vectors[0]))] for label in range(label_count)]

    for step in range(learning_count_max):
        #各クラスタの重心vectorの作成
        if (label_count==2)and(step==0):
            center_vectors[0] = [min1[0], min1[1]]
            center_vectors[1] = [max1[0], max1[1]]
        if (label_count==3)and(step==0):
            center_vectors[2] = [min1[0], min1[1]]
            center_vectors[1] = [((min1[0]+max1[0])/4), ((min1[1]+max1[1])/4)]
            center_vectors[0] = [max1[0]-1, max1[1]-1]
        #各ベクトルのラベルの再割当て
        for i, vec in enumerate(vectors):
            label_vector[i] = near(vec, center_vectors)
        #前Stepと比較し、ラベルの割り当てに変化が無かったら終了
        if old_label_vector == label_vector:
            break
        #ラベルのベクトルを保持
        old_label_vector = [l for l in label_vector]
    return center_vectors

def clustering_class3(vectors, label_count, max1, min1, learning_count_max=1000):
    """
    K-meansを行い、各ラベルの重心を返す
    :param vectors:
    :param label_count:
    :param learning_count_max:
    :return:
    """
    import random
    #各vectorに割り当てられたクラスタラベルを保持するvector
    label_vector = [random.randint(0, label_count-1) for i in vectors]
    #一つ前のStepで割り当てられたラベル。終了条件の判定に使用
    old_label_vector = list()
    #各クラスタの重心vector
    center_vectors = [[0 for i in range(len(vectors[0]))] for label in range(label_count)]

    for step in range(learning_count_max):
        #各クラスタの重心vectorの作成
        if (label_count==2)and(step==0):
            center_vectors[0] = [min1[0], min1[1]]
            center_vectors[1] = [max1[0], max1[1]]
        if (label_count==3)and(step==0):
            center_vectors[2] = [min1[0], min1[1]]
            center_vectors[1] = [((min1[0]+max1[0])/4), ((min1[1]+max1[1])/4)]
            center_vectors[0] = [max1[0]-1, max1[1]-1]
        #各ベクトルのラベルの再割当て
        for i, vec in enumerate(vectors):
            label_vector[i] = near(vec, center_vectors)
        #前Stepと比較し、ラベルの割り当てに変化が無かったら終了
        if old_label_vector == label_vector:
            break
        #ラベルのベクトルを保持
        old_label_vector = [l for l in label_vector]
    return center_vectors