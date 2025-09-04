# Implementation

## 目次
- [1. 概要](#1-概要)
- [2. データ](#2-データ)
- [3. 実装構成](#3-実装構成)
- [4. モデルと手法](#4-モデルと手法)
- [5. 評価指標](#5-評価指標)
- [6. 実行方法](#6-実行方法)
- [7. 結果のまとめ](#7-結果のまとめ)
- [8. 今後の課題](#8-今後の課題)



## 1. 概要
　本ドキュメントは、製造データを対象とした異常検知 PoC の実装内容を整理したものです。  \
　目的、全体フロー、リポジトリ構成を概要に記載します。\
　本PoCでは、複数モデルで高精度な異常検知が可能であることを確認しました。\
　詳しい評価結果は[7. 結果のまとめ](#7-結果のまとめ)に記載しています。


 ● **目的**：製造データの異常検知をPoCとして試すことを目的にこのPoCを作成しました。\
　

◆**1.1 全体フロー**◆\
 \
<img width="1015" height="690" alt="Image" src="https://github.com/user-attachments/assets/0081c7d3-a27e-4901-8713-1df53b995d9e" />
図１：全体フロー図
 

- 生データ取得：モータの<ins>電流値</ins>、<ins>速度</ins>、<ins>位置情報</ins>をPLCからCSV形式で取得
- データ前処理：ウィンドウ化や特徴量生成による整形
- モデル構築：<ins>Isolation Forest</ins>, <ins>One-Class-SVM</ins>, <ins>Autoencoder</ins>を使用
- 学習・推論：学習データでモデルを訓練し、評価データ（疑似異常を含む）で推論
- 評価：Accuracy, Precision, Recall, F1-scoreで性能を評価
  
 
◆**1.2 リポジトリ構成**◆\
　
```
PoC-Project
├─ README.md
├─ Results/
│　　├─ results_of_Autoencoder.png
│　　├─ results_of_Isolation Forest.png
│　　└─ results_of_One-Class-SVM.png
├─ data/
│　　├─ eval_features_scaled.csv　  # 評価用データ（前処理済み）
│　　└─ train_features_scaled.csv　 # 学習用データ（前処理済み）
├─ docs/
│　　└─ implementation.md　        # 実装ドキュメント
├─ notebooks/
│　　└─ PoC_walkthrough.ipynb　  # 実装の Jupyter Notebook
├─ src/
│　　├─ Autoencoder
│　　│　　├─ evaluate.py　     # Autoencoder 評価
│　　│　　└─ train.py　        # Autoencoder 学習
│　　├─ Isolation-Forest
│　　│　　├─ evaluate.py　     # Isolation Forest 評価
│　　│　　└─ train.py　        # Isolation Forest 学習
│　　├─ One-Class-SVM
│　　│　　├─ evaluate.py　     # One-Class-SVM 評価
│　　│　　└─ train.py　        # One-Class-SVM 学習
│　　├─ preprocessing
│　　│　　├─ generate-JSON.py　      # JSONファイル作成
│　　│　　├─ generate-features.py　  # 特徴量作成
│　　│　　└─normalize-features.py　  # 特徴量正規化
└─ requirements.txt         # 依存ライブラリ
　
```


---

## 2. データ
 ◆**2.1 使用データ**◆

実際にモータを使っている機械を用いたデータを収集し、このPoCに用いました。

- データ量や制約条件
  - <ins>使ったデータは製品情報になりえるため社外への情報漏洩を考慮し、公開するデータは特徴量を正規化処理したあとのデータのみに限ります。</ins>
  - 異常データは実際に起こりえるケースを想定し、スケール、ノイズ、突発ピークの調整などを行い疑似異常を作成しました。

◆**2.2 データ使用範囲**◆\
　\
![Image](https://github.com/user-attachments/assets/8a23755b-7e39-48df-94a8-a46347de660b)
図３：使用範囲イメージ図

　上記の図３は、使用したデータ範囲を示すイメージ図になります。\
　今回使用したデータ範囲は、モータの起動時と停止時を除いた運転中のみになります。\
　停止と起動をデータに含んでしまうと、停止と起動が含まれたデータだけ外れ値になってしまい、上手く学習ができませんでした。\
　なので、停止と起動を含む場合は、停止と起動を含んだ一工程ごとの計測が必要になります。\
　もしくは、停止と起動のみを切り離して処理するなどの工夫が必要になります。


- 使用データの種類
  - モータ電流値：この値をベースに推論を行います。
  - モータ現在値：JSONファイル作成時に使います。
  - モータ速度：電流値と速度の相関関係を学習するのに用います。\
　\
![Image](https://github.com/user-attachments/assets/5c6dba7c-eaa8-4531-9ed7-654a15f319ad)
図４：モータ速度種類イメージ図

　上記の図４は、使用したモータ速度パターンを示すイメージ図になります。\
　パターンは2種類になります。\
　一つは速度一定、もう一つは速度を段階的に加速させたパターンになります。\
　ただし、速度自体はバリエーションを出すためにどれも速度の違うデータを用意しました。
    
◆**2.3 前処理方法（留意点のみ抜粋）**◆

- ウィンドウ幅について
![Image](https://github.com/user-attachments/assets/092b5fe9-032f-496f-86e7-531319b584b5)
図５：周期ズレイメージ図

　上記の図５は周期についてのイメージ図になります。周期一つにつきモータがワンサイクルしているイメージです。\
　今回の実装で一番工夫した点は、データをどのように特徴量に変換するかについてです。\
　特徴量を作成するのにあたり理想は、モータの挙動ワンサイクルにつき一つの特徴量を作成をすることです。\
　そうすることで正確に正常データと異常データを比較することができます。\
　そのためには、ウィンドウ幅を実際のモータの挙動に合わせた大きさに設定する必要があります。\
　失敗例として、ウィンドウ幅を固定にしてしまうと失敗する原因に繋がります。\
　理由として、モータ速度の変化やプロットのタイミングが合わないとモータの周期にズレが発生します。\
　周期がズレると特徴量の内容に大きく影響を及ぼし、正確にワンサイクルを比較することができなくなります。
 
**下記は羅列されたデータをモータ現在値によってワンサイクルごとにウィンドウ幅を区切るスクリプトです。**
- サイクル検出スクリプト
```
# ===== サイクル列からウィンドウ検出 =====
def detect_windows(cycle_values):
    start_rows = [0]
    for i in range(1, len(cycle_values)):
        if cycle_values[i] < cycle_values[i-1]:
            start_rows.append(i)
    windows = []
    for i in range(len(start_rows)):
        start = start_rows[i]
        end = start_rows[i+1] if i+1 < len(start_rows) else len(cycle_values)
        size = end - start
        windows.append((start, end, size))
    return windows

```
 上記のスクリプトにより、ワンサイクルの始まりを設定します。\
 今回使用したデータは0〜36000000をワンサイクルとし、このサイクルが繰り返されます。\
 ひとつ前より小さくなったタイミングがサイクルの始まりとみなし「start_row(サイクル始め)」を設定します。\
 次に、設定したワンサイクルの始まりと終わり、サイズを「windows」の中にタプルとして格納します。\
 サイズは単純に終わりから始まりを引いたものになります。

 - 下記出力イメージ

   ```
   {
      CSV_win01
      "start_row": 30,
      "end_row": 91,
      "window_size": 61
   },
      CSV_win02
      "start_row": 91,
      "end_row": 153,
      "window_size": 62
   },
   ```
 
- ウィンドウフィルタリングスクリプト
```
def filter_windows(windows):
    if not windows:
        return []
    sizes = [w[2] for w in windows]
    avg_size = np.mean(sizes)
    min_size = avg_size * 0.8
    return [w for w in windows if w[2] >= min_size]
```
上記のスクリプトはウィンドウサイズにフィルタリングをかけるスクリプトです。\
データがワンサイクル分に達していないことを想定してこのスクリプトを設けました。\
内容は、ウィンドウのサイズが平均の80％を下回ると切り捨てるようになっています。\
例で言うと、0〜10までを繰り返すサイクルがあるとき、0～4までのデータが無いため5から始まってしまい、他のサイズより小さいものが出来上がってしまう場合です。\
注意点として、データの中に大幅に速度の違うデータがあるとき、データ量に差ができてしまい必要なデータまでフィルタリングされてしまう場合があります。\
今回は80％としきい値を設けましたが、この点に関しては速度ごとにしきい値を設けるなどの改善余地があるかと思います。

 - 特徴量抽出スクリプト

```
# ===== 特徴量抽出関数 =====
def extract_features(current_data, speed_data, window_size):
    feats = []
    for i in range(0, len(current_data), window_size):
        cur_win = current_data[i:i+window_size]
        spd_win = speed_data[i:i+window_size]
        if len(cur_win) == window_size and len(spd_win) == window_size:
            ratio = cur_win / np.where(spd_win == 0, 1, spd_win)  # ゼロ割防止
            feats.append({
                # 電流特徴量
                "current_mean": np.mean(cur_win),
                "current_std": np.std(cur_win),
                "current_min": np.min(cur_win),
                "current_max": np.max(cur_win),
                # 速度特徴量
                "speed_mean": np.mean(spd_win),
                "speed_std": np.std(spd_win),
                "speed_min": np.min(spd_win),
                "speed_max": np.max(spd_win),
                # 電流/速度比
                "cur_spd_ratio_mean": np.mean(ratio),
                "cur_spd_ratio_std": np.std(ratio),
                "cur_spd_ratio_min": np.min(ratio),
                "cur_spd_ratio_max": np.max(ratio),
            })
    return pd.DataFrame(feats)

```
上記は特徴量を抽出するためのスクリプトです。\
特徴量にするのは、モータの電流と速度、それと電流と速度の比を使います。\
それらをウィンドウ幅ごとに<ins>平均値</ins>、<ins>分散</ins>、<ins>最小値</ins>、<ins>最大値</ins>を算出します。\
できあがった特徴量をモデルによって学習、推論、評価といった工程に用いられます。\
特徴量の選定が悪かったり、モデルが上手く学習や推論できる形になっていないとすべてに響いてきます。\
つまり、前処理の工程がその後を左右する根幹部分になっているので、特徴量の選定や処理はできあがりを逐一確認するなど最重要項目として扱う必要があります。

---

## 3. 実装構成

- ディレクトリ構造
```
src/
├─ Autoencoder/
│   ├─ train.py        # Autoencoder 学習
│   └─ evaluate.py     # Autoencoder 評価
├─ Isolation-Forest/
│   ├─ train.py        # Isolation Forest 学習
│   └─ evaluate.py     # Isolation Forest 評価
├─ One-Class-SVM
│　　├─ evaluate.py　   # One-Class-SVM 評価
│　　└─ train.py　      # One-Class-SVM 学習
└─ preprocessing/
    ├─ generate-JSON.py       # JSONファイル作成
    ├─ generate-features.py   # 特徴量作成
    └─ normalize-features.py  # 特徴量正規化

```
### 3.1 前処理 (src/preprocessing/)

前処理工程に必要なスクリプトが用意されています。

- generate-features.py： モータ電流値・速度などのデータから特徴量を生成
- normalize-features.py： 標準化を実施し学習可能な形へ整形
- generate-JSON.py： JSON形式への変換を行い、サイクル単位で利用可能にする

### 3.2 モデル実装 (src/各モデル名/)

各モデルフォルダには train.py と evaluate.py を配置し、役割を分離しました。

- train.py： 学習処理を実行し、学習済みモデルを保存
- evaluate.py： 評価データを入力し、精度指標を算出

この構成により、異なるモデル間で同じインターフェースを用いて比較が可能です。\
また、同じ学習済みモデルで他の評価データを入力することで、再学習しなくとも評価データの比較と結果検証が可能になります。
 
### 3.3 Notebooks

- PoC_walkthrough.ipynb： 実装の流れをステップごとに実行できる形式でまとめたもの。
  
　EDA（探索的データ分析）、前処理の可視化、モデル比較の実験を含みます。\
　ディレクトリ構造の中からは外していますが、一連の実装工程が確認できるので[PoC_walkthrough.ipynb](../notebooks/PoC_walkthrough.ipynb)もご確認ください。

---

## 4. モデルと手法

本PoCでは、異常検知の手法として Isolation Forest、One-Class SVM、Autoencoder を採用しました。\
いずれも教師なし学習に基づくアプローチであり、ラベル付き異常データが限られている製造現場の課題に適している。

- **Isolation Forest**  
  Isolation Forest は、データをランダムに分割して木構造を作り、あるサンプルが「どれくらい少ない分割で孤立するか」を測定します。
　異常値は通常のデータよりも早く孤立するため、平均パス長が短くなる傾向があります。\
  Isolation Forestは計算が高速で、大規模データに適しています。\
　また、実装コストも抑えられる点も他のモデルより優れています。

特徴: 計算が高速、大規模データに適応
数式的指標: 平均パス長 $E(h(x))$ を基に異常スコアを下記式で算出

<img width="588" height="155" alt="Image" src="https://github.com/user-attachments/assets/868e30ff-2b01-4692-a783-ae4db4fa798d" />

- **One-Class SVM**  
  正常データの分布を高次元空間に写像し、境界を学習することで外れ値を判定します。\
  非線形な境界を扱えるが、データ量が多い場合は計算コストが高くなることがネックです。\

  最適化問題は以下の式で表されます。\

<img width="646" height="248" alt="Image" src="https://github.com/user-attachments/assets/fcc2f80c-34fe-4ed6-839c-ffe99420cb20" />
  


- **Autoencoder**  
  入力データを潜在空間に圧縮・復元し、再構成誤差の大きさを異常度とする。特徴量間の関係を捉えやすく、非線形な異常検知に強いです。\
  Autoencoderの算出式は下記式になります。

  <img width="583" height="112" alt="Image" src="https://github.com/user-attachments/assets/b7d2c66d-a721-440f-a3b9-16717cc3cd76" />
  

また、入力特徴量はセンサーから取得した電流値を中心に用い、標準化処理を行った上でモデルに入力しました。  \
今回使ったモデルの主要なハイパーパラメータは以下の通りです。

- 入力特徴量数：12（feature_cols）
- エンコーダ構造：入力 → 16 → 8 → 4（活性化関数は ReLU）
- デコーダ構造：4 → 8 → 16 → 出力（活性化関数は ReLU）
- 損失関数：MSELoss
- 最適化手法：Adam
- 学習率 (learning_rate)：1e-3
- エポック数 (num_epochs)：40
- バッチサイズ (batch_size)：64

◆**モデル採用理由**◆

本PoCでは、異常検知手法として Isolation Forest, One-Class SVM, Autoencoder の3種類を採用しました。\
それぞれの採用理由は以下の通りです。

- Isolation Forest
  - 汎用的かつ軽量な異常検知アルゴリズムであり、大規模データに対しても効率的に処理可能。
  - データ分布に依存せず、教師なしで異常値を分離できるため、製造現場のように異常ラベルが少ない環境に適している。

- One-Class SVM
  - 境界を明確に定義し、正常データを囲い込むアプローチにより、境界付近の挙動に敏感に反応できる。
  - 特にデータが低次元で、特徴量が適切に設計されている場合に有効。
  - 他モデルと比較して異常判定のしきい値設定を理解しやすく、比較対象として適している。

- Autoencoder
  - ニューラルネットワークを用いて入力データを再構築する仕組みのため、非線形な関係を捉えることが可能。
  - 製造データのように複雑な相関を持つ特徴量群に対して、高い表現力を発揮することが期待できる。
  - 再構築誤差に基づいた柔軟な異常検知が可能であり、深層学習ベース手法のPoCとして有効。
    
以上の3モデルを採用することで、伝統的な機械学習手法（Isolation Forest・One-Class SVM） と深層学習を活用した手法（Autoencoder）の両方を比較し、産業データにおける異常検知の有効性と特性を多面的に検証できるようにしました。 
    
---

## 5. 評価指標

- 評価用特徴量数算出スクリプト
```
import pandas as pd

# 保存した評価用特徴量を読み込み
eval_df  = pd.read_csv("eval_features.csv", encoding="utf-8-sig")

# ラベル件数をカウント
label_counts = eval_df["label"].value_counts().sort_index()

print("=== Evaluation Data Label Summary ===")
print(f"正常 (label=1)   : {label_counts.get(1, 0)} 件")
print(f"異常 (label=-1) : {label_counts.get(-1, 0)} 件")
print(f"合計             : {len(eval_df)} 件")
  ```
- 出力結果
```
=== Evaluation Data Label Summary ===
正常 (label=1)   : 612 件
異常 (label=-1) : 129 件
合計             : 741 件
```
 \
Accuracy：全体のうち、正しく分類できた割合\
Accuracy = (TP + TN) / (TP + TN + FP + FN)\
 \
Precision：異常と判定した中で、正しく異常だった割合\
Precision = TP / (TP + FP)\
 \
Recall：実際の異常のうち、正しく検出できた割合\
Recall = TP / (TP + FN)\
 \
F1-score：PrecisionとRecallのバランスをとった評価指標\
F1-score = 2 * (Precision * Recall) / (Precision + Recall)

- 使用した指標（Accuracy, Precision, Recall, F1-score）  
- 指標選定の理由  

---

## 6. 実行方法
- 環境構築（`pip install -r requirements.txt`）  
- 実行コマンド（例: Jupyter Notebook の起動方法）  

---

## 7. 結果のまとめ
![Image](https://github.com/user-attachments/assets/2b5234c7-93a9-49b5-a938-e25242dab737)\
表１：評価結果
<img width="1938" height="537" alt="Image" src="https://github.com/user-attachments/assets/7f991f7a-8d67-4782-9c37-e24e2315aeea" />
図２：評価結果グラフ\
- 評価結果（スコアの概要）  
- 考察（得られた知見、制約、注意点）
  - どのモデルも高い精度を出すことができました。詳しい結果の表やグラフは「7 結果のまとめ」をご確認ください。
- グラフを確認すると、どれだけ分離できているかがわかります。特に、Autoencoderは正常と異常をはっきりと分離されていることが視認できます。

---

## 8. 今後の課題
- データ量拡大の必要性  
- 特徴量設計の改善  
- 実運用に向けた追加検証ポイント  

