# 麻雀AI Transformer - Mahjong AI Transformer (PyTorch)

PyTorch + Transformer による**打牌判断専用**麻雀AIモデルです。  
教師あり学習（Supervised Learning）と強化学習（REINFORCEアルゴリズム）を交互に繰り返して精度を高めていきます。

---

## 特徴 / Features

- 14枚の手牌から打牌を予測（1枚を出力）
- 8層 Transformer（LayerNorm + Dropout）
- 教師あり学習と REINFORCE による強化学習
- `--resume` による中断再開機能
- 和了率・役の出現統計も取得可能

---


## 追記

import mahjong_common as mjc
import mahjong_loader as mjl

これら2つは他の方のgithubに搭載されています。
許可が得られた場合は引用させていただきます。

