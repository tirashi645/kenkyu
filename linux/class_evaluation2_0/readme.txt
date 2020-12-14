各種ソースコードについて
・connect_python.py	それぞれのソースコードを繋げる。
・Make_wavedata.py	各特徴点の動きをoptical flowを用いて検出する。
・clusteringPoint.py	手動で特徴点を分類する。
・make_figure.py	特徴点のベクトルを波形にして分類ごとに保存する。
・make_fft.py		特徴点のベクトル波形をFFT変換して図を保存する。
・make_video.py		特徴点と付きのビデオを保存する。

最初に読み込むビデオは周囲5ピクセル分パディング処理しています。