# Model Performance Analysis

## Overall Performance

The fine-tuned **ResNet-50 model achieved strong overall performance** on the evaluation dataset, reaching an accuracy of **95.31%** and correctly classifying **1,525 out of 1,600 images**. This checkpoint corresponds to **Epoch 5**, the last epoch whose weights were saved — validation loss increased in Epochs 6 and 7, so training effectively plateaued at this point.

| Metric               |      Score |
| -------------------- | ---------: |
| **Validation Loss**  |     0.2902 |
| **Accuracy**         | **95.31%** |
| **Macro Precision**  |     0.9570 |
| **Macro Recall**     |     0.9531 |
| **Macro F1-Score**   |     0.9522 |
| **Macro AUC-ROC**    | **0.9906** |

The close agreement between macro precision, recall, and F1-score indicates that the model performs relatively consistently across the four classes. Additionally, the **AUC-ROC score of 0.9906** demonstrates excellent overall class separability based on the model's predicted probabilities.

---

## Per-Class Performance

| Class       |  Precision |     Recall |   F1-Score | Samples |
| ----------- | ---------: | ---------: | ---------: | ------: |
| **Class 0** | **1.0000** |     0.8300 |     0.9071 |     400 |
| **Class 1** |     0.8986 | **0.9975** |     0.9455 |     400 |
| **Class 2** |     0.9318 |     0.9900 |     0.9600 |     400 |
| **Class 3** | **0.9975** | **0.9950** | **0.9962** |     400 |

### Class 0

Class 0 remains the model's main area of weakness. Its **precision is perfect at 100.00%**, but its **recall is only 83.00%**.

This means that whenever the model predicts Class 0, it is always correct. However, the model fails to identify a meaningful number of actual Class 0 samples.

The confusion matrix shows that:

* **39 Class 0 images** were predicted as Class 1.
* **29 Class 0 images** were predicted as Class 2.

Therefore, **68 out of 400 Class 0 samples** were misclassified. The main challenge for the model is still distinguishing Class 0 from Classes 1 and 2.

### Class 1

Class 1 achieves an outstanding **recall of 99.75%**, meaning that nearly all actual Class 1 images are correctly identified.

However, its precision is lower at **89.86%**, primarily because a significant number of Class 0 images are incorrectly predicted as Class 1 (with a handful of stray Class 2 and Class 3 samples also landing here).

This suggests that the model is highly sensitive to Class 1 but occasionally assigns the Class 1 label to samples belonging to other classes.

### Class 2

Class 2 demonstrates strong and balanced performance, with:

* **93.18% precision**
* **99.00% recall**
* **96.00% F1-score**

Only a small number of actual Class 2 samples are misclassified, with most predictions correctly assigned to their true class.

### Class 3

Class 3 is the best-performing category, achieving an F1-score of **99.62%**.

Out of 400 samples:

* **398 were correctly classified**
* Only **2 samples were predicted as Class 1**

This indicates that Class 3 has a highly distinguishable visual representation within this dataset.

---

## Confusion Matrix Analysis

The confusion matrix reveals a clear pattern in the model's errors:

```text
                Predicted
              C0    C1    C2    C3
Actual C0    332    39    29     0
Actual C1      0   399     0     1
Actual C2      0     4   396     0
Actual C3      0     2     0   398
```

The majority of predictions lie along the main diagonal, confirming the model's strong classification performance.

The most significant source of error is:

> **Class 0 being confused with Class 1 and Class 2.**

In contrast, confusion involving Class 3 is extremely limited, and Class 1/Class 2 are almost entirely free of cross-contamination with Class 0 in the reverse direction.

This pattern suggests that the model has learned strong representations for Classes 1, 2, and especially 3, but the visual characteristics of Class 0 overlap more substantially with other categories in the learned feature space.

---

## Training Trajectory

Validation loss dropped sharply in the first two epochs (1.4374 → 0.3598 → 0.3399) and continued improving steadily through **Epoch 5 (0.2902)**, alongside rising accuracy (91.63% → 95.31%). Epochs 6 and 7 both showed **increased validation loss** (0.3652 and 0.3611) despite falling training loss, a sign of overfitting beyond Epoch 5 — which is why that checkpoint was kept as the final model rather than the last epoch trained.

---

## Key Findings

* The model achieves a strong overall accuracy of **95.31%**.
* **Class 3 is classified almost perfectly**, with an F1-score of **99.62%**.
* **Classes 1 and 2 also demonstrate strong performance**, particularly in recall.
* The primary weakness is **Class 0 recall**, which falls to **83.00%**.
* Most classification errors originate from actual Class 0 samples being predicted as **Class 1 or Class 2**.
* The **AUC-ROC of 0.9906** indicates excellent overall ability to separate the four classes.
* Validation loss began rising after Epoch 5, indicating the onset of overfitting in later epochs.

## Conclusion

Overall, the fine-tuned ResNet-50 model demonstrates **strong performance on the current evaluation dataset**, with highly consistent results across most classes. The main opportunity for improvement is increasing the recall of Class 0 and reducing its confusion with Classes 1 and 2.

Possible future improvements include further investigating the Class 0 samples, examining dataset quality and class-specific visual overlap, tuning the augmentation strategy, adjusting the fine-tuning procedure or learning-rate schedule to curb the overfitting seen after Epoch 5, or using techniques designed to focus training on difficult or frequently confused examples.

While these results are promising, strong performance on this test dataset alone is **not sufficient to establish clinical reliability**. Further validation on independent and representative datasets would be necessary before making real-world medical claims.
