# ファイルの構成
WikiWithSenti が本研究のモデル
WS-BERT が従来研究のモデル

Data: 
  Pstance: P-Stanceのデータセットとwikipediaの情報がが入っている
  VAST: VASTのデータセットとwikipediaの情報がが入っている

Results:
  結果の出力場所. nohupに出力する場合はnohup.outを見て

src:
  datasets.py: データをモデルへ入力
  engine.py: 学習と評価. Line 81, 82 で学習終了条件のコードがある. ある程度F1値が上がらない場合に学習終了.
  models.py: ゲーティングメカニズムなどをするところ (λが行列)
  train.py: 学習設定
  correctmodel.py: 学会発表後にゲーティングメカニズムが上手く導入できていなかったことに気づき, 修正したもの(λが一つの値)がここには書かれている

# Combining Knowledge from Wikipedia and Sentiment Information for Stance Detection
Wikipediaと感情分析を利用したスタンス検出

## Dataset Preparation
[PStance](https://aclanthology.org/2021.findings-acl.208/), 
[COVID19-Stance](https://aclanthology.org/2021.acl-long.127/) 利用していない
[VAST](https://aclanthology.org/2020.emnlp-main.717.pdf). 

1. <em>VAST</em> [この](https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST)データセットが従来研究などで一番利用されている

2. <em>Pstance</em> Google Driveからデータセットをダウンロードした後、<em>data/pstance</em>というフォルダーに入れ, Jupiter Notebookでデータセットの前処理を行う. preprocessing.pyをして, データの前処理を行う
3. <em>COVID19-Stance</em>このデータセットは利用不可能


## Installation
[Pytorch](https://pytorch.org/get-started/locally/) と [Huggingface Transformers](https://huggingface.co/docs/transformers/installation)をインストール

## Run
PStance, 固有ターゲットのスタンス検出, Biden
```angular2html
python run_pstance_biden.py
```

PStance, クロスターゲット, Biden $\rightarrow$ Sanders
```angular2html
python run_pstance_biden2sanders.py
```
VAST, ゼロショット/数ショットのスタンス検出
```angular2html
python run_vast.py
```

## Citation
```angular2html
Issei Matsumoto, Akihiro Tamura, Kato Tsuneo:
Combining knowledge from Wikipedia and sentiment information for stance detection
The 86th National Convention of IPSJ (IPSJ2024)

```


