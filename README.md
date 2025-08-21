# Detect-anomaly-with-Isolation-Forest-in-manufacturing

## 概要

・製造現場で使われる実際のモータの電流値から、Isolation Forestを使った異常検出を行いました。\
　また、他のモデルとの比較検証を行い、実際に現場で使う際のモデルの選定について考察しました。

## アプローチ

![Image](https://github.com/user-attachments/assets/b0e1ec4f-1a4d-4411-9d11-e624cb735d91)


・サーボモータが動作したときの電流値のデータをPLCから取得しました。\
　取得したデータから前処理を施し、各モデルに適応しやすいように行います。

## 結果
（比較グラフ + 簡単な要約）
![Image](https://github.com/user-attachments/assets/dc46bfc4-4351-42f6-be50-8806cd795a5d)

・結果はIsolation ForestのF1-scoreが0.9886まで精度を上げることができました。\
　また、チューニングにより「見逃し件数」を0にしつつ、「Accuracy」を0.9885という
　高さを保ちました。\
　他のモデルと比較すると、他のモデルは「precision」は高かったが「見逃し件数」が多く
　上手く異常を検知できていないと言えます。

 
※Autoencoder(1)と(2)はエポック数が違います。(1)はエポック数30、(2)は120となっています。\
　エポック数を上げるとIsolation Forestを上回るF1-scoreが出ていますが、データが少ない環境でエポック数を上げ過ぎると学習データに\
　対し過学習してしまい、汎用性がさがってしまいます。\
　また、学習に計算コストが高くついてしまうのでAutoencoderを使う場面は検討が必要です。




## 今後の展望
（短期〜長期の展望）

## 環境
（Pythonと主要ライブラリ）

## 著者
（名前やGitHubハンドル）
