import numpy as np
np.random.seed(123)

def near(x, mu, m):
    '''特徴ベクトルの各クラスタへの帰属度を返す'''
    return [np.sum([(np.linalg.norm(x-mu[k]) / np.linalg.norm(x-mu[j])) ** (2/(m-1)) for j in range(mu.shape[0])]) ** -1 for k in range(mu.shape[0])]

def clustering(X, K=3, m=1.5):
    '''
    X: 特徴ベクトルの集合
    K: クラスタ数
    m: Fuzzyパラメータ
    '''
    # データの総数
    N = len(X)
    # 前ステップの重心
    old_mu = np.array([K, N])
    # 寄与度
    u = np.random.random([K, N])
    e = np.inf
    while True:
        # 重心の算出
        mu = np.dot(u**m, X) / np.array([np.sum(u**m, axis=1)]).T
        # 寄与度の更新
        for k in range(K):
            for i in range(N):
                u[k, i] = np.sum([(np.linalg.norm(X[i]-mu[k]) / np.linalg.norm(X[i]-mu[j]))**(2/(m-1)) for j in range(K)]) ** -1
        print(mu, old_mu)
        _e = np.linalg.norm(mu-old_mu)
        print(_e, '\n-------------------------------')
        if _e > e:break
        e = _e
    #print(mu)
    return mu

if __name__=='__main__':
    # 学習するベクトル数
    N = 100
    # 特徴ベクトルの集合
    X = np.random.random([N, 2])
    # Fuzzyパラメータ
    m = 2.0
    # クラスタリング
    mu = clustering(X, K=3, m=m)
    # 各クラスタへの所属度を求める
    near([0.1, 0.3], mu, m)