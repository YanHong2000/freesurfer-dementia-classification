# 基於 FreeSurfer 腦結構量化特徵與臨床資料之失智症分類研究

**Dementia Classification Using FreeSurfer-Derived Brain Structural Quantitative Features and Clinical Data**  
*Feature Selection, Multi-Model Comparison, and Ensemble Learning*

> 國立中山大學應用數學系碩士論文研究之 GitHub portfolio 版本。  
> 本 repository 以**研究流程、方法設計、關鍵圖表與彙整結果**為主，不公開受試者層級臨床資料、MRI 影像與 FreeSurfer 個別輸出。

## 專案概述

失智症與輕度認知障礙的臨床評估主要依賴認知量表、病史與功能狀態；結構式 MRI 則能提供臨床量表較難直接反映的腦部形態資訊。本研究以高雄榮民總醫院提供之 **3D T1-weighted MRI** 與臨床資料為基礎，透過 **FreeSurfer 7.4.1** 萃取腦區體積、皮質厚度與表面積等量化特徵，建立多任務、多模型與多階段的機器學習分析流程。

研究核心不只比較「哪個模型分數最高」，而是進一步回答：**不同 FreeSurfer 特徵組合是否具有任務差異？eTIV 校正是否有效？影像特徵是否能在完整臨床資訊之外提供穩定增量？校準、閾值與不同融合策略是否能改變結論？**

### 關鍵成果一覽

- 最終納入 **378 位受試者**：Dementia 85、MCI 233、Normal 60。
- 建立 **3 項二元分類任務**，正式分析採 **20 seeds × 5-fold stratified cross-validation**。
- FreeSurfer 初始整理 **9 個特徵群組、1,131 個量化特徵**；主分析母表經清理後包含 **397 個影像特徵與 33 個臨床特徵**。
- 前期篩選 **49 個不重複影像方向**，再進行多模型比較、ANOVA F-score 排序與 Top-N 特徵數選擇。
- 多模型比較中，**ElasticNet Logistic Regression、RidgeClassifier 等線性模型整體較穩定**；**TabPFN 多具有較高 AUC 與較佳機率品質**。
- 純影像特徵具有獨立判別力，但在完整 Clin33 基準下，加入影像後的 **AUC 增量整體有限，且未形成跨模型一致的正向提升**。

---

## 1. 研究資料與分類任務

資料涵蓋 Dementia、MCI 與 Normal 三類受試者。原始臨床資料除 ID 與標籤外共有 35 個欄位；移除收案日期與可能造成 label leakage 的 CDR score 後，保留 33 個正式臨床特徵。臨床特徵涵蓋人口學、生活型態、身體功能與健康指標，以及 MMSE、CASI 等認知量表。

| 類別 | 人數 | 比例 |
|---|---:|---:|
| Dementia | 85 | 22.5% |
| MCI | 233 | 61.6% |
| Normal | 60 | 15.9% |
| **合計** | **378** | **100%** |

三項二元分類任務如下：

| 任務 | 分類目標 | 正類 | 負類 | 樣本數 |
|---|---|---|---|---:|
| Task 1 | Dementia vs. NonDementia | Dementia (85) | MCI + Normal (293) | 378 |
| Task 2 | AbNormal vs. Normal | Dementia + MCI (318) | Normal (60) | 378 |
| Task 3 | MCI vs. Normal | MCI (233) | Normal (60) | 293 |

---

## 2. FreeSurfer 腦結構量化特徵

3D T1 MRI 經 `recon-all -all` 完成標準結構處理，並以 `segmentHA_T1.sh` 取得海馬亞區與杏仁核子核之細部分區。研究共整理 15 個 FreeSurfer 統計檔為 9 個特徵群組，初始共 1,131 個量化特徵。

| 群組 | 特徵數 | 主要資訊 |
|---|---:|---|
| brainvol | 16 | 全腦體積摘要 |
| aseg | 46 | 皮質下結構與腦室體積，含 eTIV |
| wmparc | 70 | 白質分區體積 |
| hippo | 44 | 海馬亞區體積 |
| amyg | 20 | 杏仁核子核體積 |
| aparc | 208 | Desikan–Killiany 皮質分區 |
| a2009s | 448 | Destrieux 皮質分區 |
| dkt | 193 | DKT 皮質分區 |
| ba_thresh | 86 | Brodmann area 相關分區 |

正式母表經樣本、常數、高頻率值與重複特徵清理後，保留 **378 位受試者、397 個 FreeSurfer 特徵與 33 個臨床特徵**。需要由資料估計的缺失值填補、標準化與類別編碼均在各外層訓練折內估計後再套用至測試折，以降低資料洩漏風險。

---

## 3. 驗證設計與模型

正式模型評估採 **20 個隨機種子 × 5 折分層交叉驗證**，同一任務下不同資料方向沿用相同外層切分。對需要超參數選擇的模型，每個外層訓練折內再進行 3-fold 分層內層交叉驗證；同一 seed 的 5 個外層測試折合併為完整 out-of-fold (OOF) 預測，供後續校準、閾值與集成分析使用。

研究各階段共涉及 13 種基礎分類模型：

- **線性模型**：Logistic Regression、ElasticNet Logistic Regression、RidgeClassifier、LinearSVC
- **樹與梯度提升**：RandomForest、ExtraTrees、XGBoost、LightGBM、CatBoost
- **核與距離方法**：RBF-SVC、kNN
- **機率生成模型**：Gaussian Naive Bayes
- **表格式基礎模型**：TabPFN

主要評估指標涵蓋 **Balanced Accuracy、ROC-AUC、Sensitivity、Specificity、Brier score、ECE**，並以 Wilcoxon signed-rank test 與 BH-FDR 檢視重複資料切分下 AUC 差異方向的一致性。

---

## 4. 整體研究流程

![整體研究架構與分析流程](figures/01_research_workflow.png)

整體分析分為兩條主線：

1. **前期管線**：影像方向篩選 → eTIV 校正比較 → 多模型基線 → ANOVA F-score 特徵排序 → Top-N → Phase 1–4。
2. **完整臨床基準下之影像增量分析**：保留 Clin33，獨立排序 FreeSurfer 特徵形成 FS_topK，進一步比較純臨床、純影像、早期融合、中期融合與晚期融合。

---

# 主要研究結果

## 5. FreeSurfer 影像方向篩選

前期使用 ElasticNet Logistic Regression 比較 **9 個單一群組、10 個具醫學意義組合與 30 個延伸組合，共 49 個不重複方向**。三項任務的優勢影像訊號並不完全相同：Task 1 較受益於皮質分區、皮質下與內側顳葉結構的整合；Task 2 與 Task 3 則更集中於海馬與皮質下結構。

| 任務 | 後續主要方向 | 正式 Top-N |
|---|---|---:|
| Task 1 | APARC_ASEG_AMYG | Top-80 |
| Task 1 | APARC_ASEG_HIPPO | Top-80 |
| Task 2 | HIPPO_ASEG | Top-30 |
| Task 3 | HIPPO | Top-20 |

三項任務在代表方向下的全特徵多模型基線如下。整體而言，**ElasticNet Logistic Regression 在三項代表方向皆取得最高或接近最高 BalAcc；TabPFN 則持續呈現較高 AUC**。

<details>
<summary><strong>展開：三項任務之多模型基線圖</strong></summary>

### Task 1 - APARC_ASEG_AMYG

![Task 1 baseline](figures/03_baseline_task1.png)

### Task 2 - HIPPO_ASEG

![Task 2 baseline](figures/04_baseline_task2.png)

### Task 3 - HIPPO

![Task 3 baseline](figures/05_baseline_task3.png)

</details>

**研究判讀：** Task 1 與 Task 2 的整體表現高於 Task 3；MCI vs. Normal 的區辨較困難。更重要的是，模型的 AUC、機率品質與固定操作點下的 BalAcc 並不完全一致，因此後續不能只依單一指標決定模型。

---

## 6. eTIV 校正：統計關聯不等於預測效能提升

在清理後六個入選 FreeSurfer 群組中，共辨識 256 個候選體積特徵；以 Normal 子群進行線性關聯檢定後，**201 個特徵在 BH-FDR 校正後仍與 eTIV 顯著相關（78.5%）**。研究再以折內殘差法建立 Adj 與 No_Adj 版本比較分類表現。

結果顯示：**eTIV 校正沒有帶來一致的 BalAcc 或 AUC 改善**。Task 1 大致持平，Task 2 與 Task 3 多數方向略為下降，因此後續主要分析統一採用 **No_Adj**。

這個結果說明：即使腦區體積與顱內總體積具有統計上顯著的線性關係，移除該變異也不一定會轉化為更好的分類表現。

---

## 7. 特徵排序與 Top-N 選擇

研究使用 **ANOVA F-score** 對候選方向中的臨床與影像特徵共同排序，再以 ElasticNet Logistic Regression 與 RidgeClassifier 掃描不同 Top-N 特徵子集。

![Top-N Balanced Accuracy](figures/06_topn_balanced_accuracy.png)

Task 1 兩個主要方向最終統一採用 **Top-80**；Task 2 採 **Top-30**；Task 3 兩個模型均於 **Top-20** 取得最高 BalAcc，因此採 Top-20。整體未呈現「特徵越多，表現越好」的趨勢。

前段排序主要由認知量表主導，但影像特徵亦呈現一致的神經解剖分布：

- Task 1：除 CASI、MMSE 外，影像特徵包括**左側楔前葉、左側海馬、中顳回、下顳回、杏仁核子核**。
- Task 2 / Task 3：較集中於**左側全海馬、海馬體部與 GC-ML-DG、molecular layer、CA3 等海馬亞區**。

這表示 FreeSurfer 特徵的主要價值不只在分類分數，也在於其能提供與認知退化相關的神經解剖解釋。

<details>
<summary><strong>展開：Top-N AUC 補充圖</strong></summary>

![Top-N AUC](figures/11_topn_auc_appendix.png)

</details>

---

## 8. Phase 1：正式基線

完成方向、模型與特徵數篩選後，Phase 1 以 OOF 預測比較臨床加影像方向與純臨床方向。

| 任務 | 純臨床最高 BalAcc | 純臨床最高 AUC | 最佳 clin_fs BalAcc | 最佳 clin_fs AUC |
|---|---|---|---|---|
| Task 1 | RidgeClassifier **0.848** | TabPFN **0.928** | LinearSVC 0.841 | TabPFN 0.919 |
| Task 2 | RidgeClassifier **0.853** | TabPFN **0.932** | RidgeClassifier 0.839 | TabPFN 0.932 |
| Task 3 | RidgeClassifier **0.810** | TabPFN **0.905** | ElasticNet Logistic 0.808 | TabPFN 0.903 |

**研究判讀：** Task 1 與 Task 2 的純臨床方向呈現最穩定優勢；Task 3 的 clin_fs 在部分模型或指標上接近或略有改善，但未形成跨模型一致的影像增益。TabPFN 多具有較高 AUC 與較低 Brier score，但固定閾值下的 BalAcc 未必最高；線性模型則較常維持 Sensitivity 與 Specificity 的平衡。

---

## 9. Phase 2：Probability Calibration

Phase 2 比較 Raw、Platt scaling 與 Isotonic regression。七個資料方向的平均結果呈現相同趨勢：**Platt scaling 在所有方向取得最低平均 Brier score；Isotonic regression 則在所有方向取得最低平均 ECE**。

![Probability calibration](figures/07_probability_calibration.png)

以 52 組「任務 × 方向 × 模型」設定計算，Platt scaling 在 38 組取得最低 Brier score；ECE 則全部 52 組均以 Isotonic regression 最低。Isotonic 的分段常數映射可能壓縮樣本排序，因此其 AUC 略降；Platt 能改善機率品質並保留 Raw 的 AUC，因此後續主要閾值分析採用 **Platt scaling**。

---

## 10. Phase 3：決策閾值與操作點

固定閾值 0.5 在三項任務產生明顯不同的 Sensitivity / Specificity 配置。以純臨床方向為例：

| 任務 | 0.5 BalAcc | Youden BalAcc | 0.5 Sens / Spec | Youden Sens / Spec |
|---|---:|---:|---|---|
| Task 1 | 0.770 | **0.832** | 0.611 / 0.929 | 0.841 / 0.823 |
| Task 2 | 0.727 | **0.824** | 0.954 / 0.500 | 0.794 / 0.854 |
| Task 3 | 0.719 | **0.787** | 0.941 / 0.496 | 0.780 / 0.795 |

Task 1 的適合閾值明顯低於 0.5，而 Task 2、Task 3 則高於 0.5。Youden 在目前資料下較能平衡兩類辨識能力；Sens@0.90 與 Spec@0.90 則可依篩檢或避免誤判需求，主動調整錯誤型態。

**研究判讀：** AUC 衡量的是排序能力，最終分類行為仍受到操作閾值影響。固定 0.5 並不是三項任務都合理的決策基準。

---

## 11. Phase 4：Ensemble Learning

研究比較 equal weighting、performance weighting 與 convex Super Learner (convexSL)，並區分同一資料方向內的模型集成與跨方向預測集成。

方向內集成的 ΔBalAcc 範圍為 **-0.0077 至 +0.0031**，只有部分方向得到小幅改善；跨方向集成的六種組合則**全部未超越各自最佳單一方向**，ΔBalAcc 介於 **-0.0054 至 -0.0002**。

**研究判讀：** 集成並沒有自動帶來更高效能。當基礎模型或資料方向本身高度相關時，增加組合數量不代表能取得真正的誤差互補。

---

# 完整臨床基準下的影像增量分析

## 12. 重新建立 Clin33 / FS_topK / Clin33_FS_topK

前期管線中的 clin_fs 會將臨床與影像特徵共同排序，因此無法直接回答「完整臨床資訊都保留後，MRI 還有沒有額外價值」。因此研究重新建立：

- **Clin33**：完整 33 個正式臨床預測特徵
- **FS_topK**：只對 397 個 FreeSurfer 特徵排序後取 Top-K
- **Clin33_FS_topK**：完整臨床 + Top-K 影像（early fusion）
- **NN_fusion**：臨床與影像雙分支的 intermediate fusion
- **Late fusion**：不同資料方向的 OOF 預測於輸出層整合

影像 Top-K 以五種代表模型的純影像 AUC 綜合評估，最終固定為 **Task 1 Top-30、Task 2 Top-30、Task 3 Top-40**。

![FS Top-K AUC](figures/08_fs_topk_auc.png)

![FS Top-K group composition](figures/09_fs_topk_group_composition.png)

純影像高排名特徵仍主要集中於海馬、杏仁核、皮質下結構與部分皮質區域，與前期方向篩選及特徵排序所觀察到的神經解剖訊號一致。

---

## 13. Early / Intermediate Fusion 的主要分類表現

| 任務 | 設定 | 代表模型 | BalAcc | AUC |
|---|---|---|---:|---:|
| Task 1 | Clin33 | RidgeClassifier | **0.843** | 0.922 |
|  | FS_top30 | GaussianNB | 0.705 | 0.772 |
|  | Clin33_FS_top30 | TabPFN | 0.842 | **0.929** |
|  | NN_fusion | NN_fusion | 0.806 | 0.909 |
| Task 2 | Clin33 | RidgeClassifier | **0.844** | **0.928** |
|  | FS_top30 | GaussianNB | 0.711 | 0.782 |
|  | Clin33_FS_top30 | ElasticNet Logistic | 0.834 | 0.926 |
|  | NN_fusion | NN_fusion | 0.774 | 0.888 |
| Task 3 | Clin33 | ExtraTrees | **0.803** | **0.898** |
|  | FS_top40 | ElasticNet Logistic | 0.691 | 0.746 |
|  | Clin33_FS_top40 | TabPFN | 0.802 | **0.898** |
|  | NN_fusion | NN_fusion | 0.748 | 0.852 |

純影像方向在三項任務皆明顯低於 Clin33。Early fusion 能接近 Clin33，Task 1 甚至取得較高 AUC，但 BalAcc 未超越純臨床；Task 2、Task 3 也沒有形成明確整體優勢。Intermediate fusion 的 NN_fusion 在三項任務均低於 Clin33 與 early fusion。

### NN_fusion 架構

![NN fusion](figures/02_nn_fusion_architecture.png)

NN_fusion 先由臨床與影像分支分別學習 16 維潛在表徵，再進行 concatenation 與分類。結果顯示，在目前樣本數與表格式區域特徵下，增加多模態網路複雜度並沒有自然轉化成更高效能。

---

## 14. 核心問題：MRI 在完整臨床基準之外有沒有穩定增量？

以相同 12 個模型逐一比較 `Clin33_FS_topK - Clin33` 的 AUC，三項任務共形成 36 項主要比較。

![Imaging incremental delta AUC](figures/10_imaging_increment_delta_auc.png)

| 任務 | 比較 | q<0.05 且 ΔAUC>0 | q<0.05 且 ΔAUC<0 | 平均 ΔAUC |
|---|---|---:|---:|---:|
| Task 1 | Clin33_FS_top30 vs. Clin33 | **0** | 8 | -0.0053 |
| Task 2 | Clin33_FS_top30 vs. Clin33 | **0** | 12 | -0.0126 |
| Task 3 | Clin33_FS_top40 vs. Clin33 | **0** | 9 | -0.0148 |

三項任務**都沒有出現 q<0.05 且平均 ΔAUC 為正的模型**。純影像相對 Clin33 的平均 AUC 差距更大，三項任務約為 -0.144 至 -0.151。

> **主要結論：** FreeSurfer 結構影像特徵具有獨立分類訊號，也提供具有神經解剖意義的補充資訊；但在本研究的完整 Clin33 基準下，影像加入後並未形成跨模型、跨任務一致且穩定的額外排序能力。

這項結果是本研究最重要的研究判斷之一：**「影像有訊號」與「影像能在完整臨床模型上增加預測價值」是兩個不同問題。**

---

## 15. 不同融合層級比較

各格為 `BalAcc / AUC`：

| 任務 | Clin33 | FS_topK | Early fusion | Intermediate fusion | Late fusion |
|---|---|---|---|---|---|
| Task 1 | **0.843 / 0.922** | 0.705 / 0.772 | 0.842 / **0.929** | 0.806 / 0.909 | 0.842 / 0.922 |
| Task 2 | **0.844 / 0.928** | 0.711 / 0.782 | 0.834 / 0.926 | 0.774 / 0.888 | 0.836 / 0.923 |
| Task 3 | 0.803 / **0.898** | 0.691 / 0.746 | 0.802 / **0.898** | 0.748 / 0.852 | **0.805** / 0.895 |

Early fusion 與 late fusion 均優於 intermediate fusion，但兩者也沒有形成跨任務一致優於 Clin33 的結果。Task 3 的 late fusion BalAcc 由 0.803 微幅提升至 0.805，但 AUC 反而由 0.898 降至 0.895，表示這項改善只出現在特定操作點，並未伴隨排序能力同步提升。

<details>
<summary><strong>展開：convexSL 晚期融合方向權重</strong></summary>

![convexSL direction weights](figures/12_convexsl_direction_weights.png)

</details>

---

## 16. 固定超參數敏感度分析

為檢視結果是否高度依賴逐外層折調參，本研究另以各設定在 100 個外層折中最常出現的超參數組合重新執行主要分析。固定參數與逐折調參版本在 **AUC 等排序能力指標上整體相近**；BalAcc、Sensitivity 與 Specificity 等閾值相關指標的波動稍大，但主要模型排序、影像增量方向與最佳晚期融合設定大致維持一致。

因此，研究主要結論並非由單一超參數選擇方式所驅動。

---

# 主要研究結論

1. **不同認知分類任務具有不同的影像訊號。** Task 1 較需要皮質、皮質下與內側顳葉資訊整合；Task 2 / 3 則更集中於海馬及相關結構。
2. **較多影像特徵不等於較好。** 49 個候選方向與 Top-N 掃描皆顯示，擴大特徵範圍並未帶來一致改善。
3. **eTIV 校正未改善預測表現。** 多數體積特徵雖與 eTIV 顯著相關，但殘差校正沒有穩定轉化為分類增益。
4. **臨床特徵是目前最強且較穩定的預測來源。** 線性模型的分類平衡較穩定，TabPFN 則常具有較高 AUC 與較佳機率品質。
5. **MRI 的主要價值較偏向補充性與解釋性，而非穩定增量。** FreeSurfer 特徵本身能分類，也呈現海馬、杏仁核與皮質區域等神經解剖訊號；但在完整 Clin33 基準上，其額外 AUC 增益有限。
6. **模型複雜度與融合複雜度提高，不保證效能提升。** NN_fusion 與多數 ensemble / late-fusion 設定沒有形成穩定優勢。

---

## 研究限制

- **樣本規模與外部驗證有限**：高維影像特徵與複雜模型的穩定性仍可能受有限樣本與類別不平衡影響。
- **影像表徵仍以 FreeSurfer 區域統計量為主**：尚未完整利用原始 MRI、surface mesh、label 或其他影像模態。
- **eTIV 僅評估一種殘差校正策略**：比例校正、非線性與多變量校正仍可進一步比較。
- **融合設計相對受控**：未評估更複雜的注意力或多模態 representation learning。
- **部分研究層級決策未完全巢狀於外層 CV**：方向、模型與部分特徵選擇先於後續正式分析固定，因此結果較適合解讀為系統性比較與穩定性分析，而非完全獨立的端到端泛化估計。
- **重複 seeds 使用同一批受試者**：Wilcoxon 與 BH-FDR 主要用於描述不同切分下差異方向的一致程度，不視為獨立受試者層級的母體推論。

---

## Repository Structure

```text
freesurfer-dementia-classification/
│
├── README.md
├── .gitignore
│
├── figures/
│   ├── 01_research_workflow.png
│   ├── 02_nn_fusion_architecture.png
│   ├── 03_baseline_task1.png
│   ├── 04_baseline_task2.png
│   ├── 05_baseline_task3.png
│   ├── 06_topn_balanced_accuracy.png
│   ├── 07_probability_calibration.png
│   ├── 08_fs_topk_auc.png
│   ├── 09_fs_topk_group_composition.png
│   ├── 10_imaging_increment_delta_auc.png
│   ├── 11_topn_auc_appendix.png
│   └── 12_convexsl_direction_weights.png
│
├── tables/
│   └── 01_...csv ~ 16_...csv
│
└── docs/
    └── master_thesis.pdf
```

`figures/` 收錄 README 使用的主要研究流程與結果圖；`tables/` 提供由最終論文整理出的主要結果 CSV。README 僅保留作品集需要的核心分析與結論，完整模型設定、逐模型結果、統計檢定與附錄內容仍以正式論文為準。

---

## Data Availability

本研究使用之臨床資料與 3D T1 MRI 影像由高雄榮民總醫院提供，涉及人體研究資料與資料使用限制，因此本 repository 不公開受試者層級的臨床原始資料、MRI / DICOM 影像、FreeSurfer 個體輸出、個體層級預測與中間分析資料。

本 repository 僅提供研究方法、彙整後統計結果、研究圖表與完整論文，作為研究成果與分析流程之展示。

---

## 使用工具與環境

- **Neuroimaging**：FreeSurfer 7.4.1 (`recon-all -all`, `segmentHA_T1.sh`)
- **OS for FreeSurfer processing**：Ubuntu 22.04.5 LTS
- **Machine Learning / Statistics**：Logistic / ElasticNet / Ridge / SVM、RandomForest / ExtraTrees、XGBoost、LightGBM、CatBoost、kNN、Gaussian Naive Bayes、TabPFN
- **Multimodal modeling**：Early fusion、NN_fusion intermediate fusion、late fusion、convex Super Learner
- **Evaluation**：Nested / repeated stratified cross-validation、OOF prediction、ANOVA F-score、probability calibration、threshold optimization、ensemble learning、Wilcoxon signed-rank test、BH-FDR
- **Documentation**：LaTeX

---

## 完整論文

完整研究方法、逐模型結果、統計檢定、附錄與參考文獻請見：

**[Master's Thesis PDF](docs/master_thesis.pdf)**

論文題目：**基於 FreeSurfer 腦結構量化特徵與臨床資料之失智症分類研究：特徵篩選、多模型比較與集成學習**  
National Sun Yat-sen University, Department of Applied Mathematics, 2026.
