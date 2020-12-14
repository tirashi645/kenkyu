分類するプログラム（すべて共通）

connect_python.pyを実行すれば手動分類，アルゴリズムでの分類，画像の保存を行います．

clusteringPoint.py	手動分類するプログラム（分類アルゴリズムとは関係ないプログラムです）
Make_wavedata.py	アルゴリズムを使って自動分類するプログラム
save_pict.py		分類結果を出力，保存するプログラム


Make_wavedataでは，class2_outputとclass_output3の中でアルゴリズムを使って分類しています．
class1は変更していないのでk-meansで実行しています．

clusteringPoint.pyでは最初に画像が表示されるので　画像を右クリック→×ボタン　で終了してください（評価するときに使うプログラム）

保存データはグラフがD:/~/plt/class(1,2,3)/に入るようになっており画像がD:/~/resultに入るようになっています．

使用している動画はsample_videoに入っています．
とりあえず1_Trinm.mp4をベースに動作確認しています．