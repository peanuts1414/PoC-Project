# Implementation

## 1. 概要
本ドキュメントは、製造データを対象とした異常検知 PoC の実装内容を整理したものです。  
目的、リポジトリ構成、使用データ、実装方法、評価結果の概要を記載します。

- **目的**：製造データの異常検知をPoCとして試すことを目的にこのPoCを作成しました。
　　
- **リポジトリ構成**\
PoC-Project\
├─ READNE.md\
├─ Results/\
│　　├─ results of Autoencoder.png\
│　　├─ results of Isolation Forest.png\
│　　└─ results of One-Class-SVM.png\
├─ data/\
│　　├─ eval_features_scaled.csv　  # 元データ\
│　　└─ train_features_scaled.csv　 # 前処理済みデータ\
├─ docs/\
│　　└─ implementation.md　        # 本ドキュメント\
├─ notebooks/\
│　　└─ PoC_walkthrough.ipynb　  # 実装に用いたスクリプト\
├─ src/\
│　　├─ Autoencoder\
|　　|　　├─ evaluate.py　     # 評価スクリプト\
|　　|　　└─ train.py　        # Autoencoder 学習\
│　　├─ Isolation-Forest\
|　　|　　├─ evaluate.py　     # 評価スクリプト\
|　　|　　└─ train.py　        # Isolation Forest 学習\
│　　├─ One-Class-SVM\
|　　|　　├─ evaluate.py　     # 評価スクリプト\
|　　|　　└─ train.py　        # One-Class-SVM 学習\
│　　├─ preprocessing\
|　　|　　├─ generate-JSON.py　      # JSONファイル作成スクリプト\
|　　|　　├─ generate-features.py　  # 特徴量作成スクリプト\
|　　|　　└─normalize-features.py　  # 特徴量正規化スクリプト\
└─ requirements.txt         # 依存ライブラリ

  

<img width="1398" height="950" alt="Image" src="https://github.com/user-attachments/assets/0e07e116-24f9-4695-b5ba-fa0c902680d3" />

---

## 2. データ
- 使用データの種類  
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

