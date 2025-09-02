# Implementation

## 1. 概要
本ドキュメントは、製造データを対象とした異常検知 PoC の実装内容を整理したものです。  
目的、リポジトリ構成、使用データ、実装方法、評価結果の概要を記載します。

 ● **目的**：製造データの異常検知をPoCとして試すことを目的にこのPoCを作成しました。\
　\
◆**リポジトリ構成**◆\
　\
PoC-Project\
├─ README.md\
├─ Results/\
│　　├─ results_of_Autoencoder.png\
│　　├─ results_of_Isolation Forest.png\
│　　└─ results_of_One-Class-SVM.png\
├─ data/\
│　　├─ eval_features_scaled.csv　  # 評価用データ（前処理済み）\
│　　└─ train_features_scaled.csv　 # 学習用データ（前処理済み）\
├─ docs/\
│　　└─ implementation.md　        # 実装ドキュメント\
├─ notebooks/\
│　　└─ PoC_walkthrough.ipynb　  # 実装の Jupyter Notebook\
├─ src/\
│　　├─ Autoencoder\
│　　│　　├─ evaluate.py　     # Autoencoder 評価\
│　　│　　└─ train.py　        # Autoencoder 学習\
│　　├─ Isolation-Forest\
│　　│　　├─ evaluate.py　     # Isolation Forest 評価\
│　　│　　└─ train.py　        # Isolation Forest 学習\
│　　├─ One-Class-SVM\
│　　│　　├─ evaluate.py　     # One-Class-SVM 評価\
│　　│　　└─ train.py　        # One-Class-SVM 学習\
│　　├─ preprocessing\
│　　│　　├─ generate-JSON.py　      # JSONファイル作成\
│　　│　　├─ generate-features.py　  # 特徴量作成\
│　　│　　└─normalize-features.py　  # 特徴量正規化\
└─ requirements.txt         # 依存ライブラリ\
　\
 ◆**使用データ**◆
- モータ電流値
- モータ速度
- モータ現在値

◆**実装方法**◆\
 \
<img width="1398" height="950" alt="Image" src="https://github.com/user-attachments/assets/0e07e116-24f9-4695-b5ba-fa0c902680d3" />

- 生データ：モータの電流値、速度、現在値をPLCからCSVデータで取得
- データ前処理：データの整形（ウィンドウ化、特徴量生成）
- モデル構築：Isolation Forest, One Class SVM, Autoencoderを使用
- 学習：モデルが学習用の特徴量を基に学習
- 推論：モデルが評価用の疑似異常データを含んだデータで、正常値と異常値を分離し判定
- 評価指標：Accuracy,Precision,Recall,F1-scoreを用いて評価\
　\
◆**評価結果**◆\
　\
![Image](https://github.com/user-attachments/assets/2b5234c7-93a9-49b5-a938-e25242dab737)
<img width="1938" height="537" alt="Image" src="https://github.com/user-attachments/assets/7f991f7a-8d67-4782-9c37-e24e2315aeea" />
  
- 評価結果の表を見るとどのモデルも高い精度を出せているとわかります。表からの評価だけでは選定しがたいです。
- グラフを確認すると、どれだけ分離できているかがわかります。特に、Autoencoderは正常と異常をはっきりと分離されていることが視認できます。


---

## 2. データ
◆**データ使用範囲**◆\
　\
![Image](https://github.com/user-attachments/assets/8a23755b-7e39-48df-94a8-a46347de660b)

今回使用したデータ範囲は、モータの起動時と停止時を除いた運転中のみになります。\
停止と起動をデータに含んでしまうと、停止と起動が含まれたデータだけ外れ値になってしまい、上手く学習ができませんでした。\
なので、停止と起動を含む場合は、停止と起動を含んだ一工程ごとの計測が必要になります。\
もしくは、停止と起動のみを切り離して処理するなどの工夫が必要になります。


- 使用データの種類
  - モータ電流値：この値をベースに推論を行います。
  - モータ現在値：JSONファイル作成時に使います。
  - モータ速度：電流値と速度の相関関係を学習するのに用います。
    　
    　 
    
- 前処理方法（標準化、欠損値処理、ラベル付け 等）  
- データ量や制約条件  

---

## 3. 実装構成
- ディレクトリ構造（例: `src/`, `notebooks/`, `data/`）  
- 各ファイルの役割  

---

## 4. モデルと手法
- 使用モデル（Isolation Forest, One-Class SVM, Autoencoder）  
- モデルの概要説明  
- モデル選定の理由  

---

## 5. 評価指標
- 使用した指標（Accuracy, Precision, Recall, F1-score）  
- 指標選定の理由  

---

## 6. 実行方法
- 環境構築（`pip install -r requirements.txt`）  
- 実行コマンド（例: Jupyter Notebook の起動方法）  

---

## 7. 結果のまとめ
- 評価結果（スコアの概要）  
- 考察（得られた知見、制約、注意点）  

---

## 8. 今後の課題
- データ量拡大の必要性  
- 特徴量設計の改善  
- 実運用に向けた追加検証ポイント  

